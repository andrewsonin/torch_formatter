from abc import abstractmethod
from collections import abc
from os import PathLike
from pathlib import Path
from time import time
from typing import Iterable, Optional, Tuple, Callable, Generator, TypeVar, List, Union

import matplotlib.pyplot as plt
import torch
from IPython.display import clear_output
from matplotlib.ticker import MaxNLocator

__all__ = (
    'BatchIterator',
    'TorchTrainer'
)

_mpl_integer_locator = MaxNLocator(integer=True)

Batch = TypeVar('Batch')
FwdResult = TypeVar('FwdResult')
BatchIterable = Iterable[Batch]


class BatchIterator:
    __slots__ = ()

    def __iter__(self) -> Generator[Batch, None, None]:
        return self.batch_generator()

    @abstractmethod
    def batch_generator(self) -> Generator[Batch, None, None]:
        pass


class TorchTrainer:
    __slots__ = (
        # Main
        '_network',
        '_optimizer',
        '_train_function',
        '_loss_function',

        # Data iterators
        '_train_iterator',
        '_test_iterator',
        '_valid_iterator',

        # Constants
        '_n_epochs',
        '_clip_rate',

        # Loss history
        '_valid_loss_history',
        '_train_loss_history',
        '_test_loss_history',

        # Auxiliary info
        '_epoch',
        '_completed',

        # Timing attributes
        '_time_elapsed',

        # Draw settings
        '_alpha',
        '_figsize',
        '_train_loss_color',
        '_valid_loss_color',
        '_test_loss_color',
        '_savefig_path'
    )

    def __init__(
            self,
            network: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            *,
            train_function: Callable[['TorchTrainer', Batch], FwdResult],
            loss_function: Callable[['TorchTrainer', Batch, FwdResult], torch.FloatTensor],
            train_iterator: BatchIterable,
            test_iterator: Optional[BatchIterable] = None,
            valid_iterator: Optional[BatchIterable] = None,
            n_epochs: int,
            clip_rate: Optional[float] = None,
            alpha: float = 0.97,
            figsize: Tuple[float, float] = (9, 6),
            train_loss_color: Union[str, float] = 'b',
            valid_loss_color: Union[str, float] = 'r',
            test_loss_color: Union[str, float] = 'g',
            savefig_path: Optional[Union[PathLike, str, Path]] = None
    ) -> None:

        if not isinstance(network, torch.nn.Module):
            raise TypeError('Parameter `network` should be an instance of type `torch.nn.Module`')

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('Parameter `optimizer` should be an instance of type `torch.optim.Optimizer`')

        if not isinstance(train_function, abc.Callable):
            raise TypeError('Parameter `train_function` should be callable')

        if not isinstance(loss_function, abc.Callable):
            raise TypeError('Parameter `loss_function` should be callable')

        if not isinstance(train_iterator, BatchIterator):
            raise TypeError('Parameter `train_iterator` should be an instance of `BatchIterator`')

        if not isinstance(test_iterator, BatchIterator) and test_iterator is not None:
            raise TypeError('Parameter `test_iterator` should be an instance of `BatchIterator` or None')

        if not isinstance(valid_iterator, BatchIterator) and valid_iterator is not None:
            raise TypeError('Parameter `valid_iterator` should be an instance of `BatchIterator` or None')

        if type(n_epochs) is not int:
            raise TypeError('Parameter `n_epochs` should be of type `int`')

        if not isinstance(clip_rate, (int, float)) and clip_rate is not None:
            raise TypeError('Parameter `clip_rate` should be of type `int`, `float` or None')

        if not isinstance(alpha, (int, float)):
            raise TypeError('Parameter `alpha` should be of type `int` or `float`')

        if (
                type(figsize) is not tuple
                or len(figsize) != 2
                or not all(isinstance(cord, (int, float)) for cord in figsize)
        ):
            raise TypeError('Parameter `figsize` should be a `tuple` with two `int` or `float` elements')

        if not isinstance(savefig_path, (Path, str, PathLike)) and savefig_path is not None:
            raise TypeError(
                'Parameter `savefig_path` should be an instance of either PathLike object or `str`, or None'
            )

        self._network = network
        self._optimizer = optimizer
        self._loss_function = loss_function
        self._train_function = train_function
        self._train_iterator = train_iterator
        self._test_iterator = test_iterator

        self._valid_iterator = valid_iterator
        self._n_epochs = n_epochs
        self._clip_rate = clip_rate

        self._alpha = alpha
        self._figsize = figsize
        self._train_loss_color = train_loss_color
        self._valid_loss_color = valid_loss_color
        self._test_loss_color = test_loss_color
        self._savefig_path = savefig_path

        self._valid_loss_history = []
        self._train_loss_history = []
        self._test_loss_history = []
        self._epoch = 1
        self._completed = False
        self._time_elapsed = 0

    def forward(self, batch: Batch) -> torch.FloatTensor:
        return self._train_function(self, batch)

    def calc_loss(self, batch: Batch, train_result: torch.FloatTensor) -> torch.FloatTensor:
        return self._loss_function(self, batch, train_result)

    @property
    def network(self) -> torch.nn.Module:
        return self._network

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def train_iterator(self) -> BatchIterable:
        return self._train_iterator

    @property
    def test_iterator(self) -> Optional[BatchIterable]:
        return self._test_iterator

    @property
    def valid_iterator(self) -> Optional[BatchIterable]:
        return self._valid_iterator

    @property
    def n_epochs(self) -> int:
        return self._n_epochs

    @property
    def clip_rate(self) -> float:
        return self._clip_rate

    @property
    def train_loss_history(self) -> List[float]:
        return self._train_loss_history.copy()

    @property
    def test_loss_history(self) -> List[float]:
        return self._test_loss_history.copy()

    @property
    def valid_loss_history(self) -> List[float]:
        return self._valid_loss_history.copy()

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def time_elapsed(self) -> float:
        return self._time_elapsed

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def figsize(self) -> Tuple[float, float]:
        return self._figsize

    @property
    def train_loss_color(self) -> Union[str, float]:
        return self._train_loss_color

    @property
    def test_loss_color(self) -> Union[str, float]:
        return self._test_loss_color

    @property
    def valid_loss_color(self) -> Union[str, float]:
        return self._valid_loss_color

    def clear_history(self) -> None:
        self._valid_loss_history = []
        self._train_loss_history = []
        self._test_loss_history = []
        self._epoch = 1
        self._completed = False
        self._time_elapsed = 0

    def train(self) -> None:

        # >>> Loading variables onto the stack >>>
        train_loss_history = self._train_loss_history
        valid_loss_history = self._valid_loss_history
        test_loss_history = self._test_loss_history

        network = self._network
        optimizer = self._optimizer

        train_iterator = self._train_iterator
        valid_iterator = self._valid_iterator
        test_iterator = self._test_iterator

        forward = self.forward
        calc_loss = self.calc_loss

        clip_rate = self._clip_rate

        n_epochs = self._n_epochs

        alpha = self._alpha
        figsize = self._figsize
        train_loss_color = self._train_loss_color
        valid_loss_color = self._valid_loss_color
        test_loss_color = self._test_loss_color
        savefig_path = self._savefig_path
        # <<< Loading variables onto the stack <<<

        time_stamp_1 = time()

        for epoch in range(self._epoch, n_epochs + 1):

            train_loss = 0
            network.train(True)

            for n_batch, train_batch in enumerate(train_iterator, 1):

                optimizer.zero_grad()

                result = forward(train_batch)
                loss = calc_loss(train_batch, result)

                loss.backward()

                if clip_rate is not None:
                    torch.nn.utils.clip_grad_norm_(network.parameters(), clip_rate)

                optimizer.step()
                train_loss += loss.item()

            train_loss /= n_batch

            test_loss = val_loss = 0.0
            network.train(False)

            with torch.no_grad():
                if valid_iterator is not None:
                    for n_batch, valid_batch in enumerate(valid_iterator, 1):
                        result = forward(valid_batch)
                        loss = calc_loss(valid_batch, result)
                        val_loss += loss
                    val_loss /= n_batch

                if test_iterator is not None:
                    for n_batch, test_batch in enumerate(test_iterator, 1):
                        result = forward(test_batch)
                        loss = calc_loss(test_batch, result)
                        test_loss += loss
                    test_loss /= n_batch

            # >>> Safe statistics storage >>>
            try:

                train_loss_history.append(train_loss)
                if valid_iterator is not None:
                    valid_loss_history.append(val_loss)
                if test_iterator is not None:
                    test_loss_history.append(test_loss)

            except KeyboardInterrupt:

                if len(train_loss_history) != epoch:
                    train_loss_history.append(train_loss)
                if (
                        valid_iterator is not None
                        and len(valid_loss_history) < len(train_loss_history)
                ):
                    valid_loss_history.append(val_loss)
                if (
                        test_iterator is not None
                        and len(test_loss_history) < len(valid_loss_history)
                ):
                    test_loss_history.append(test_loss)

                raise

            finally:

                self._epoch += 1

                time_stamp_2 = time()
                time_diff = time_stamp_2 - time_stamp_1
                time_stamp_1 = time_stamp_2
                self._time_elapsed += time_diff
                speed = 1 / time_diff
            # <<< Safe statistics storage <<<

            # >>> Plotting >>>
            _cur_epoch_range = range(1, epoch + 1)

            fig = plt.figure(figsize=figsize)
            fig.gca().xaxis.set_major_locator(_mpl_integer_locator)
            plt.plot(
                _cur_epoch_range,
                train_loss_history,
                label='Train',
                color=train_loss_color,
                alpha=alpha
            )
            if valid_iterator is not None:
                plt.plot(
                    _cur_epoch_range,
                    valid_loss_history,
                    label='Valid',
                    color=valid_loss_color,
                    alpha=alpha
                )
            if test_iterator is not None:
                plt.plot(
                    _cur_epoch_range,
                    test_loss_history,
                    label='Test',
                    color=test_loss_color,
                    alpha=alpha
                )
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()
            if savefig_path is not None:
                plt.savefig(savefig_path)

            clear_output(True)
            print(
                f'Epoch:        {epoch:10_}\n'

                f'Train loss:   {train_loss:10.6}      '
                f'Val loss:       {val_loss:10.6}      '
                f'Test loss: {test_loss:10.6}\n'

                f'Time elapsed: {int(self._time_elapsed):10_} sec  '
                f'Time remaining: {int((n_epochs - epoch) * speed):10_} sec  '
                f'Speed:     {speed:10.6} ep/sec'
            )
            plt.show()
            # <<< Plotting <<<

        self._completed = True

