"""
A model worker that executes the model.
"""
from InstructorEmbedding import INSTRUCTOR
import numpy as np
from typing import List

import asyncio
from queue import Queue
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
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
        prompt = requests_dict["prompts"]
        instruction = [requests_dict["instructions"]] * len(prompt)

        return [list(x) for x in zip(instruction, prompt)]

    @staticmethod
    def numpy_array_to_list(array):
        return array.tolist()


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


class ModelWorker:
    """
    init a model worker
    """

    def __init__(self, model_path: str, limit_worker_concurrency: int = 1):
        """
        Parameters:
        model_path (str): path/to/model/dir
        """
        self.model_path = model_path
        self.model = None

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


print()


@app.post("/worker_get_embeddings")
async def api_get_embeddings(request: Request):
    params = await request.json()
    await queue_line.acquire_worker_semaphore()
    params = HelperFunction.unpack_instruction(params)
    embedding = worker.compute_embeddings(params)
    embedding = HelperFunction.numpy_array_to_list(embedding)
    queue_line.release_worker_semaphore()
    return JSONResponse(content=embedding)


if __name__ == "__main__":
    worker = ModelWorker(model_path="instructor-xl")
    queue_line = TrafficControl(limit_worker_concurrency=2)
    worker.load_model()
    uvicorn.run(app, host="0.0.0.0", port=1234)
