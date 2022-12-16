from abc import ABC, abstractmethod

from tensorflow._api.v2.v2.keras import Model


class InterfaceModelUtilities(ABC):
    @abstractmethod
    def model_architecture(self) -> Model:
        pass

    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    def sampling_rate(self) -> int:
        pass

    @abstractmethod
    def need_3D_input(self) -> bool:
        pass
