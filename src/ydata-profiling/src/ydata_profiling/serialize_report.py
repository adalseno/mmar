import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from ydata_profiling.profile_report import ProfileReport

from ydata_profiling.config import Settings
from ydata_profiling.model import BaseDescription
from ydata_profiling.report.presentation.core import Root
from ydata_profiling.version import __version__


class SerializeReport:
    """Extend the report to be able to dump and load reports."""

    df = None
    config = None
    _df_hash: Optional[str] = None
    _report = None
    _description_set = None

    @property
    def df_hash(self) -> Optional[str]:
        return None

    def dumps(self) -> bytes:
        """
        Serialize ProfileReport and return bytes for reproducing ProfileReport or Caching.

        Returns:
            Bytes which contains hash of DataFrame, config, _description_set and _report
        """
        import pickle

        # Note: _description_set and _report may are None if they haven't been computed
        return pickle.dumps(
            [
                self.df_hash,
                self.config,
                self._description_set,
                self._report,
            ]
        )

    def loads(self, data: bytes) -> Union["ProfileReport", "SerializeReport"]:
        """
        Deserialize the serialized report

        Args:
            data: The bytes of a serialize ProfileReport object.

        Raises:
            ValueError: if ignore_config is set to False and the configs do not match.

        Returns:
            self
        """
        import pickle

        try:
            (
                df_hash,
                loaded_config,
                loaded_description_set,
                loaded_report,
            ) = pickle.loads(data)
        except Exception as e:
            raise ValueError("Failed to load data") from e

        if not all(
            (
                df_hash is None or isinstance(df_hash, str),
                isinstance(loaded_config, Settings),
                loaded_description_set is None
                or isinstance(loaded_description_set, BaseDescription),
                loaded_report is None or isinstance(loaded_report, Root),
            )
        ):
            raise ValueError(
                "Failed to load data: file may be damaged or from an incompatible version"
            )
        if (df_hash == self.df_hash) or (self.df is None):
            # load to an empty ProfileReport
            # Set description_set, report, sample if they are None，or raise an warning.
            if self._description_set is None:
                self._description_set = loaded_description_set
            else:
                warnings.warn(
                    "The description set of current ProfileReport is not None. It won't be loaded."
                )
            if self._report is None:
                self._report = loaded_report
            else:
                warnings.warn(
                    "The report of current ProfileReport is not None. It won't be loaded."
                )

            # overwrite config
            self.config = loaded_config

            # warn if version not equal
            if (
                loaded_description_set is not None
                and loaded_description_set.package["ydata_profiling_version"]
                != __version__
            ):
                warnings.warn(
                    f"The package version specified in the loaded data is not equal to the version installed. "
                    f"Currently running on ydata-profiling {__version__} , while loaded data is generated by ydata_profiling, {loaded_description_set.package['ydata_profiling_version']}."
                )

            # set df_hash
            self._df_hash = df_hash

        else:
            raise ValueError("DataFrame does not match with the current ProfileReport.")
        return self

    def dump(self, output_file: Union[Path, str]) -> None:
        """
        Dump ProfileReport to file
        """
        if not isinstance(output_file, Path):
            output_file = Path(str(output_file))

        output_file = output_file.with_suffix(".pp")
        output_file.write_bytes(self.dumps())

    def load(
        self, load_file: Union[Path, str]
    ) -> Union["ProfileReport", "SerializeReport"]:
        """
        Load ProfileReport from file

        Raises:
             ValueError: if the DataFrame or Config do not match with the current ProfileReport
        """
        if not isinstance(load_file, Path):
            load_file = Path(str(load_file))

        self.loads(load_file.read_bytes())
        return self
