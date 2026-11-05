# Task 1 - Few-shot classification as attention over a support set

**Python / PyTorch coding exercise | Suggested time: 35-45 minutes**

[Download the candidate handout (PDF)](../output/pdf/task-1-few-shot-classification.pdf)
or start in [`task1_matching.py`](task1_matching.py).

Implement a few-shot classifier that uses attention over labelled support
examples to classify queries. This exercise focuses on indexing, matrix
operations, numerical stability, and algorithmic reasoning.

## Episode inputs

| Input | Shape / type | Meaning |
| --- | --- | --- |
| `support_embeddings` | Float tensor `[N_s, D]` | Support feature vectors |
| `support_labels` | Long tensor `[N_s]` | Support class IDs in `[0, C)` |
| `query_embeddings` | Float tensor `[N_q, D]` | Query feature vectors |
| `query_labels` | Long tensor `[N_q]` | Query target class IDs in `[0, C)` |
| `num_classes` | Integer `C` | Total number of classes |
| `temperature` | Float, default `0.1` | Positive attention temperature |

## For every query

1. Compute cosine similarity to every support example.
2. Divide similarities by `temperature`.
3. Apply softmax over the support examples.
4. Obtain class probabilities by summing the attention weights of support
   examples belonging to each class.
5. Return the mean negative log-likelihood of the query labels and the predicted
   classes.

## Implement

```python
def matching_loss(
    support_embeddings,
    support_labels,
    query_embeddings,
    query_labels,
    num_classes,
    temperature=0.1,
):
    ...
```

Return `(loss, predictions)`: a scalar mean negative log-likelihood and a
`LongTensor[N_q]` containing the highest-probability class for each query.
Keep the loss differentiable with respect to both embedding tensors.

## Requirements and assumptions

- Do not assume examples are grouped by class or that classes have equal
  numbers of support examples.
- Use vectorized PyTorch operations and numerically stable computations.
  Treat a zero embedding as a zero normalized vector.
- Assume non-empty support and query sets, a shared positive embedding dimension,
  finite floating-point embeddings of the same dtype, tensors on the same
  device, valid `int64` labels, `num_classes > 0`, and a finite positive
  temperature. You do not need to implement input validation.
- Every query's target class occurs in the support set. Other classes may be
  absent; they receive zero probability.

Be ready to explain your tensor shapes, numerical choices, time and space
complexity, and the tests you would use. No training or dataset download is needed.
