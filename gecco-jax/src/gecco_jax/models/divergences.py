from torch_dimcheck import dimchecked, A


@dimchecked
def mse(xs: A["N* D"], ys: A["N* D"]) -> A[""]:
    return ((xs - ys) ** 2).mean()
