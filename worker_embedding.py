"""
A model worker that executes the model.
"""
from InstructorEmbedding import INSTRUCTOR
import numpy as np
import pandas as pd
from typing import List
import shutil
import os
from datetime import datetime
import io
import ast

import asyncio
from queue import Queue
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()


class HelperFunction:
    """
    Function collection to handle model I/O
    """

    @staticmethod
    def cosine_similarity(A: np.array, B: np.array) -> np.array:
        """
        Calculate cosine similarity between each row of A and B.

        Parameters:
        A (numpy array of shape [n, m]): First input array.
        B (numpy array of shape [w, m]): Second input array.

        Returns:
        numpy array of shape [n, w]: Cosine similarity matrix.
        """
        # Normalize the vectors to unit length (L2 normalization)
        A_normalized = A / np.linalg.norm(A, axis=1, keepdims=True)
        B_normalized = B / np.linalg.norm(B, axis=1, keepdims=True)

        # Calculate the dot product between normalized vectors
        dot_product = np.dot(A_normalized, B_normalized.T)

        return dot_product

    @staticmethod
    def unpack_instruction(requests_dict):
        # broadcast / append instruction for each prompt
        prompt = requests_dict["prompts"]
        instruction = [requests_dict["instructions"]] * len(prompt)

        return [list(x) for x in zip(instruction, prompt)]

    @staticmethod
    def numpy_array_to_list(array):
        return array.tolist()

    @staticmethod
    def create_backup(filename):
        # Get the current date in the format "YYYY-MM-DD"
        current_date = datetime.today()

        # Check if the file exists
        if os.path.exists(filename):
            # Create the backup filename with the format "fileA.txt.bak.YYYY-MM-DD"
            backup_filename = f"{current_date}_{filename}.bak"

            # Copy the original file to the backup filename
            shutil.copy(filename, backup_filename)
            print(f"Backup created: {backup_filename}")
        else:
            print(f"no existing cache exist in directory. no backup cache created")

    @staticmethod
    def convert_pd_list_column_to_numpy_array():
        pass


class TrafficControl:
    """
    limit queue traffic
    """

    def __init__(self, limit_worker_concurrency: int = 1):
        self.limit_worker_concurrency = limit_worker_concurrency
        self.semaphore = None

    def release_worker_semaphore(self):
        self.semaphore.release()

    def acquire_worker_semaphore(self):
        if self.semaphore is None:
            self.semaphore = asyncio.Semaphore(self.limit_worker_concurrency)
        return self.semaphore.acquire()

    def get_queue_length(self):
        if (
            self.semaphore is None
            or self.semaphore._value is None
            or self.semaphore._waiters is None
        ):
            return 0
        else:
            return (
                self.limit_worker_concurrency
                - self.semaphore._value
                + len(self.semaphore._waiters)
            )


class ModelWorker:
    """
    init a model worker
    """

    def __init__(
        self,
        model_path: str,
        csv_cache_dir: str = "embeddings_cache.parquet",
        limit_worker_concurrency: int = 1,
    ):
        """
        Parameters:
        model_path (str): path/to/model/dir
        """
        self.model_path = model_path
        self.model = None
        self.csv_cache_dir = csv_cache_dir
        # the column should be ["embeddings", "instructions", "prompts"]
        if os.path.exists(csv_cache_dir):
            print("loading csv from cache")
            self.df = self._load_cache()
        else:
            print(
                "no cache detected please load the cache manually or using API endpoint to create new one"
            )
            self.df = None

    def load_model(self, load_to: str = None) -> object:
        print(self)
        """
        pin model to memory
        """
        self.model = INSTRUCTOR(self.model_path)
        # pin to gpu mem
        if load_to != None:
            self.model.to(load_to)

    def compute_embeddings(self, nested_text_list: List[List[str]]) -> np.array:
        """
        compute latent embedding
        """

        return self.model.encode(nested_text_list)

    def compute_cosine_sim(
        self, nested_text_list: List[List[str]], top_k: int = 5, min_prob: float = 0.60
    ) -> np.array:
        """
        calculate how similar or related the question withe the cached answer. it will return probability for each answer.
        use top_k (default to 5) to retrieve high prob answer only or set it to -1 to get all of sample
        """
        question_embedding = self.model.encode(nested_text_list)
        # convert pandas column filled with 1d np.array to numpy 2d np.array
        answer_embedding = np.stack(self.df["embeddings"].to_list())
        # compute how close the question related to the answer
        similarity_score = HelperFunction.cosine_similarity(
            question_embedding, answer_embedding
        )
        # Get the indices that would sort the array in descending order
        sorted_indices_descending = np.argsort(-similarity_score)
        # grab N highest value
        # [:] all first dimension
        # [:,:5] all first dimension and 5 second dimension
        top_k_index = sorted_indices_descending[:, :top_k]

        # top closest answer in the dataframe

        top_list = []

        for count, top_answer_index in enumerate(top_k_index):
            # copy so it wont mess the original df
            top_answer_df = self.df.prompts.iloc[top_answer_index].copy()
            top_answer_df = pd.DataFrame(top_answer_df)
            top_answer_df = top_answer_df.rename(columns={"prompts": "answer"})
            # store score prob alongside the answer
            top_answer_df["score"] = similarity_score[count, top_answer_index]
            # drop the answer that has low probability
            top_answer_df = top_answer_df[top_answer_df["score"] >= min_prob]
            # convert to dict so it's usable as json output
            top_answer_dict = top_answer_df.to_dict(orient="records")

            top_list.append(top_answer_dict)
        return top_list

    def compute_df_embeddings(self) -> None:
        """
        compute all embeddings inside dataframe
        """
        list_prompt = [
            list(x) for x in zip(self.df["instructions"], self.df["prompts"])
        ]

        embeddings = self.model.encode(list_prompt)
        # convert 2d np array to a list of 1d np.array
        list_of_embeddings = [x for x in embeddings]
        self.df["embeddings"] = list_of_embeddings

    def create_dataframe_from_csv(self, csv: str) -> None:
        """
        reads csv and create worker dataframe for further process
        """
        df = pd.read_csv(csv)

        # check if all the necessary column exist
        if not all(col in df.columns for col in ["instructions", "prompts"]):
            raise Exception("instruction / prompt column not detected")
        self.df = df

    def overwrite_latent_cache(self) -> None:
        """
        this method overwrite csv cache, the old cache is saved
        """
        # backup cache file
        HelperFunction.create_backup(self.csv_cache_dir)
        cache = self.df.copy()
        # cache["embeddings"] = cache["embeddings"].apply(lambda x: x.tolist())
        cache.to_parquet(self.csv_cache_dir)
        del cache

    def _load_cache(self) -> pd.DataFrame:
        df = pd.read_parquet(self.csv_cache_dir)
        # df["embeddings"] = df["embeddings"].apply(lambda x: np.array(ast.literal_eval(x)))
        return df


print()


@app.post("/get_embeddings")
async def api_get_embeddings(request: Request):
    """
    This API endpoint allows you to obtain embeddings for a given prompt.
    The embeddings are computed based on the provided prompt and an instruction.

    Parameters:
    ----------
    request : Request
    - The incoming HTTP request containing the JSON body with the following structure:
    --------
        {
            'prompt': [list_of_string],
            'instruction': str
        }
    - 'prompt' (list of strings): The list of strings representing the prompt for which embeddings need to be computed.
    - 'instruction' (string): An instruction string that may be used to guide the worker's behavior while computing the embeddings.

    Returns:
    --------
    JSONResponse
        If the request is successful, the API will return a JSON response with the computed embeddings.
        The response will be a list of floating-point numbers representing the embeddings.

    Examples:
    ---------
    Request:
    -------
        POST /worker_get_embeddings
        Content-Type: application/json

        {
            "prompt": ["question 1", "question 2"],
            "instruction": "Compute embeddings for the given prompt."
        }

    Response:
    --------
        Status: 200 OK
        Content-Type: application/json
        [
            [
                0.12345,
                0.67891,
                -0.98765,
                ...
            ],
            [
                0.12345,
                0.67891,
                -0.98765,
                ...
            ]
        ]
    """
    params = await request.json()
    await queue_line.acquire_worker_semaphore()
    params = HelperFunction.unpack_instruction(params)
    embedding = worker.compute_embeddings(params)
    embedding = HelperFunction.numpy_array_to_list(embedding)
    queue_line.release_worker_semaphore()
    return JSONResponse(content=embedding)


@app.post("/create_cache_from_csv")
async def upload_csv(file: UploadFile = File(...)):
    # Get the file contents
    contents = await file.read()

    # memory csv object
    contents = io.BytesIO(contents)
    # reads csv and store df internally
    worker.create_dataframe_from_csv(csv=contents)
    # generate embeddings for each prompt
    worker.compute_df_embeddings()
    worker.overwrite_latent_cache()

    return {"status": "embedding stored in cache"}


@app.post("/calculate_close_match_index")
async def calculate_top_k(request: Request):
    """
    This API endpoint allows you to calculate closest answer given question.
    use batched list for faster inference.

    Parameters:
    ----------
    request : Request
    - The incoming HTTP request containing the JSON body with the following structure:
    --------
        {
            'prompt': [list_of_question_string],
            'instruction': str,
            'top_k': int
        }
    - 'prompt' (list of strings): The list of strings representing the prompt for which embeddings need to be computed.
    - 'instruction' (string): An instruction string that may be used to guide the worker's behavior while computing the embeddings.
    - 'top_k' (int): n number of closest match ordered by the most likely answer to least likely answer.
    Returns:
    --------
    JSONResponse
        If the request is successful, the API will return a JSON response with the computed embeddings.
        The response will be a list of floating-point numbers representing the embeddings.

    Examples:
    ---------
    Request:
    -------
        POST /worker_get_embeddings
        Content-Type: application/json

        {
            "prompt": ["question 1", "question 2"],
            "instruction": "Compute embeddings for the given prompt."
        }

    Response:
    --------
        Status: 200 OK
        Content-Type: application/json
        [
            [
               5,
               2,
               45,
                ...
            ],
            [
                6,
                20,
                0,
                ...
            ]
        ]
    """
    params = await request.json()
    await queue_line.acquire_worker_semaphore()
    params = HelperFunction.unpack_instruction(params)
    answer = worker.compute_cosine_sim(params)
    # embedding = HelperFunction.numpy_array_to_list(embedding)
    queue_line.release_worker_semaphore()
    return JSONResponse(content=answer)


if __name__ == "__main__":
    worker = ModelWorker(model_path="instructor-xl")
    queue_line = TrafficControl(limit_worker_concurrency=2)
    worker.load_model()
    uvicorn.run(app, host="localhost", port=1234)
