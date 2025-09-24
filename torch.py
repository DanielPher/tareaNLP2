class _CUDA:
    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def device_count() -> int:
        return 0


def device(name: str):
    return name


class _Version:
    cuda = None


cuda = _CUDA()
version = _Version()
