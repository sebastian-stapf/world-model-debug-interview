"""Task 1 starter. Read README.md or the PDF for the full prompt."""


def matching_loss(
    support_embeddings,
    support_labels,
    query_embeddings,
    query_labels,
    num_classes,
    temperature=0.1,
):
    """Return (scalar mean NLL, predicted class IDs [N_q]).

    support_embeddings: floating tensor [N_s, D]
    support_labels: int64 tensor [N_s], values in [0, num_classes)
    query_embeddings: floating tensor [N_q, D]
    query_labels: int64 tensor [N_q], values in [0, num_classes)

    Compute cosine attention over support examples, aggregate attention by
    class, and use the resulting class distribution to classify each query.
    Keep the loss differentiable with respect to both embedding tensors.
    Inputs satisfy the assumptions in the handout; no validation is required.
    """
    raise NotImplementedError("Implement matching_loss for Task 1.")
