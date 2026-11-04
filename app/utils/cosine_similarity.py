import numpy as np


def cosine_similarity(
        embedding1,
        embedding2
):
    """
    Calculate cosine similarity between
    two face embeddings.
    """

    embedding1 = np.asarray(
        embedding1,
        dtype=np.float32
    )

    embedding2 = np.asarray(
        embedding2,
        dtype=np.float32
    )

    norm1 = np.linalg.norm(
        embedding1
    )

    norm2 = np.linalg.norm(
        embedding2
    )

    if (
            norm1 <= 1e-12
            or
            norm2 <= 1e-12
    ):
        return 0.0

    return float(
        np.dot(
            embedding1,
            embedding2
        )
        /
        (
                norm1 * norm2
        )
    )