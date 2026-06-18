class SimpleLossCompute:
    """A simple loss compute function that returns the loss tensor."""
    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(self, x, target, normalize_batch_size=True):
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              target.contiguous().view(-1))

        if normalize_batch_size:
            # KLDivLoss with reduction='sum' needs to be normalized by batch size.
            # For simplicity, we assume batch size can be inferred from target.
            batch_size = target.size(0)
            loss = loss / batch_size

        return loss # Return the loss tensor, not a detached item

