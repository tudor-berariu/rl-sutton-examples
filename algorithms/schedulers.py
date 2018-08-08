from typing import Union


class Schedule(object):

    def __init__(self) -> None:
        self.__crt_value = None

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError

    @property
    def crt_value(self):
        return self.__crt_value

    def __str__(self) -> str:
        raise NotImplementedError


class Constant(Schedule):

    def __init__(self, value) -> None:
        super(Constant, self).__init__()
        self.__crt_value = value

    def __next__(self):
        return self.__crt_value

    def __str__(self, precision: int = 1, scientific: bool = True) -> str:
        fmt = f".{precision:d}{'e' if scientific else 'f':s}"
        return f"{self.__crt_value:{fmt}}"


class Linear(Schedule):

    def __init__(self, start, step, end, repeat_last: bool = True) -> None:
        super(Linear, self).__init__()
        self.__start, self.__step, self.__end = start, step, end
        self.__repeat_last = repeat_last

    def __iter__(self):
        self.__crt_value = self.__start
        return self

    def __next__(self):
        step, end = self.__step, self.__end
        value = self.__crt_value
        if (value < end and step > 0) or (value > end and step < 0):
            self.__crt_value = new_value = value + step
            if (step > 0 and new_value > end) or (step < 0 and new_value < end):
                self.__crt_value = end
            return value
        elif self.__repeat_last:
            return value
        raise StopIteration

    def __str__(self, precision: int = 1, scientific: bool = True) -> str:
        fmt = f".{precision:d}{'e' if scientific else 'f':s}"
        return f"({self.__start:{fmt}}:{self.__step:{fmt}}:{self.__end:{fmt}})"


def get_schedule(name: Union[str, int, float], **kwargs) -> Schedule:
    if isinstance(name, (float, int)):
        return Constant(name)
    if name == "const":
        return Constant(**kwargs)
    elif name == "linear":
        return Linear(**kwargs)
    raise ValueError(name)


__all__ = ['get_schedule', 'Constant', 'Linear']


def example():
    # 1. LinearDecay
    kwargs = {"start": .5, "step": -.05, "end": .2}
    values = get_schedule("linear", **kwargs)
    print(f"{values}:", end="")
    for idx, val in zip(range(10), values):
        print(f" [{idx:d}] {val: .3f}", end="")
    print("")

    # 2. Constant
    values = get_schedule("const", value=.75)
    print(f"{values}:", end="")
    for _idx, val in zip(range(10), values):
        print(f" {val: .3f}", end="")
    print("")

    # 3. Concise way to call with a single value
    values = get_schedule(.75)
    print(f"{values}:", end="")
    for _idx, val in zip(range(10), values):
        print(f" {val: .3f}", end="")
    print("")


if __name__ == "__main__":
    example()
