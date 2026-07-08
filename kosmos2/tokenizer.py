
# --- Conceptual Bounding Box Tokenizer ---
class BBoxTokenizer:
    def __init__(self, num_coord_bins: int = 1000, coord_token_offset: int = 0):
        self.num_coord_bins = num_coord_bins
        self.coord_token_offset = coord_token_offset

    def _quantize_coordinate(self, coord: float) -> int:
        return min(int(coord * self.num_coord_bins), self.num_coord_bins - 1)

    def tokenize_bbox(self, bbox: list[float]) -> list[int]:
        if not all(0.0 <= c < 1.0 for c in bbox):
            raise ValueError("Bounding box coordinates must be normalized floats between 0 and 1.")

        coord_tokens = [
            self._quantize_coordinate(bbox[0]) + self.coord_token_offset,
            self._quantize_coordinate(bbox[1]) + self.coord_token_offset,
            self._quantize_coordinate(bbox[2]) + self.coord_token_offset,
            self._quantize_coordinate(bbox[3]) + self.coord_token_offset
        ]
        return coord_tokens

    def describe_bbox_tokens(self, tokens: list[int]) -> list[str]:
        return [f"<loc_{t - self.coord_token_offset:03d}>" for t in tokens]

