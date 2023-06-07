import os
import re
from datetime import datetime
from glob import glob
from typing import Callable, Iterator, List, Tuple, Union

import pyworkflow.protocol.params as params
import pyworkflow.utils as pwutils
from pyworkflow.protocol import Protocol
from pyworkflow.protocol.params import Form


class ProtImport(Protocol):
    """Base class for other all Import protocols."""


class ProtImportFiles(ProtImport):
    """Base class for other Import protocols.
    All imports protocols will have:
    1) Several options to import from (_getImportOptions function)
    2) First option will always be "from files". (for this option
      files with a given pattern will be retrieved  and the ### will
      be used to mark an ID part from the filename.
      - For each file a function to process it will be called
        (_importFile(fileName, fileId))
    """

    IMPORT_FROM_FILES = 0

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form: Form) -> None:
        form.addSection(label="Import")

        form.addParam(
            "filesPath",
            params.PathParam,
            label="Files directory",
            help="Directory with the files you want to import.\n\n"
            "The path can also contain wildcards to select"
            "from several folders. \n\n"
            "Examples:\n"
            "  ~/project/data/day??_files/\n"
            "Each '?' represents one unknown character\n\n"
            "  ~/project/data/day*_files/\n"
            "'*' represents any number of unknown characters\n\n"
            "  ~/project/data/day##_files/\n"
            "'##' represents two digits that will be used as "
            "file ID\n\n"
            "NOTE: wildcard characters ('*', '?', '#') "
            "cannot appear in the actual path.)",
        )
        form.addParam(
            "filesPattern",
            params.StringParam,
            label="Pattern",
            help="Pattern of the files to be imported.\n\n"
            "The pattern can contain standard wildcards such as\n"
            "*, ?, etc, or special ones like ### to mark some\n"
            "digits in the filename as ID.\n\n"
            "NOTE: wildcards and special characters "
            "('*', '?', '#', ':', '%') cannot appear in the "
            "actual path.",
        )
        form.addParam(
            "copyFiles",
            params.BooleanParam,
            default=False,
            expertLevel=params.LEVEL_ADVANCED,
            label="Copy files?",
            help="By default the files are not copied into the "
            "project to avoid data duplication and to save "
            "disk space. Instead of copying, symbolic links are "
            "created pointing to original files. This approach "
            "has the drawback that if the project is moved to "
            "another computer, the links need to be restored.",
        )

        self._defineImportParams(form)

        self._defineAcquisitionParams(form)

        form.addSection("Streaming")

        form.addParam(
            "dataStreaming",
            params.BooleanParam,
            default=False,
            label="Process data in streaming?",
            help="Select this option if you want import data as it is "
            "generated and process on the fly by next protocols. "
            "In this case the protocol will keep running to check "
            "new files and will update the output Set, which can "
            "be used right away by next steps.",
        )

        form.addParam(
            "timeout",
            params.IntParam,
            default=43200,
            condition="dataStreaming",
            label="Timeout (secs)",
            help="Interval of time (in seconds) after which, if no new file "
            "is detected, the protocol will end. When finished, "
            "the output Set will be closed and no more data will be "
            "added to it. \n"
            "Note 1:  The default value is  high (12 hours) to avoid "
            "the protocol finishes during the acquisition of the "
            "microscope. You can also stop it from right click and press "
            "STOP_STREAMING.\n"
            "Note 2: If you're using individual frames when importing "
            "movies, the timeout won't be refreshed until a whole "
            "movie is stacked.",
        )

        form.addParam(
            "fileTimeout",
            params.IntParam,
            default=30,
            condition="dataStreaming",
            label="File timeout (secs)",
            help="Interval of time (in seconds) after which, if a file has "
            "not changed, we consider it as a new file. \n",
        )

        self._defineBlacklistParams(form)

    def _defineImportParams(self, form: Form) -> None:
        """Override to add options related to the different types
        of import that are allowed by each protocol.
        """
        pass

    def _defineAcquisitionParams(self, form: Form) -> None:
        """Override to add options related to acquisition info."""
        pass

    def _defineBlacklistParams(self, form: Form) -> None:
        """Override to add options related to blacklist info."""
        pass

    def _getDefaultChoice(self) -> int:
        return self.IMPORT_FROM_FILES

    # --------------------------- INFO functions ------------------------------
    def _validate(self) -> List[str]:
        errors = []
        if not self.getPattern():
            errors.append("The path and pattern can not be both empty!!!")
        else:
            # Just check the number of files matching the pattern
            self.getMatchFiles()
            if self.numberOfFiles == 0:
                errors.append(
                    "There are no files matching the pattern %s" % self.getPattern()
                )

        return errors

    # --------------------------- BASE methods to be overwritten ----------------
    def _getImportChoices(self) -> List[str]:
        """Return a list of possible choices
        from which the import can be done.
        (usually packages formats such as: xmipp3, eman2, relion...etc.
        """
        return ["files"]

    # --------------------------- UTILS functions -----------------------------
    def getPattern(self) -> str:
        """Expand the pattern using environ vars or username
        and also replacing special character # by digit matching.
        """
        self._idRegex = None
        filesPath = self.filesPath.get("").strip()
        filesPattern = self.filesPattern.get("").strip()

        if filesPattern:
            fullPattern = os.path.join(filesPath, filesPattern)
        else:
            fullPattern = filesPath

        pattern = pwutils.expandPattern(fullPattern.replace("$", ""))
        match = re.match("[^#]*(#+)[^#]*", pattern)

        if match is not None:
            g = match.group(1)
            n = len(g)
            # prepare regex pattern - place ids, handle *, handle ?
            idregex = pattern.replace(g, "(%s)" % ("[0-9]" * n))
            idregex = idregex.replace("*", ".*")
            idregex = idregex.replace("?", ".")
            self._idRegex = re.compile(idregex)
            pattern = pattern.replace(g, "[0-9]" * n)

        return pattern

    def getMatchFiles(self, pattern=None) -> List[str]:
        """Return a sorted list with the paths of files that
        matched the pattern.
        """
        if pattern is None:
            pattern = self.getPattern()

        filePaths = glob(pattern)
        filePaths.sort()
        self.numberOfFiles = len(filePaths)

        return filePaths

    def getCopyOrLink(self) -> Callable:
        # Set a function to copyFile or createLink
        # depending in the user selected option
        if self.copyFiles:
            return pwutils.copyFile
        else:
            return pwutils.createAbsLink

    def fileModified(self, fileName: str, fileTimeout: float) -> bool:
        """Check if the fileName modification time is less
        than a given timeout.
        Params:
            fileName: input filename that will be checked.
            fileTimeout: timeout"""
        self.debug("Checking file: %s" % fileName)
        mTime = datetime.fromtimestamp(os.path.getmtime(fileName))
        delta = datetime.now() - mTime
        self.debug("   Modification time: %s" % pwutils.prettyTime(mTime))
        self.debug("   Delta: %s" % pwutils.prettyDelta(delta))

        return delta.seconds < fileTimeout

    def isBlacklisted(self, fileName: str) -> bool:
        """Overwrite in subclasses"""
        return False

    def iterFiles(self) -> Iterator[Tuple[str, Union[int, None]]]:
        """Iterate through the files matched with the pattern.
        Provide the fileName and fileId.
        """
        filePaths = self.getMatchFiles()

        for fileName in filePaths:
            if self._idRegex:
                # Try to match the file id from filename
                # this is set by the user by using #### format in the pattern
                match = self._idRegex.match(fileName)
                if match is None:
                    raise Exception(
                        "File '%s' doesn't match the pattern '%s'"
                        % (fileName, self.getPattern())
                    )

                fileId = int(match.group(1))

            else:
                fileId = None

            yield fileName, fileId

    @classmethod
    def worksInStreaming(cls):
        # Import protocols always work in streaming
        return True


class ProtImportFile(ProtImport):
    """Base class for other Import protocols.
    All imports protocols will have:
    1) Several options to import from (_getImportOptions function)
    2) First option will always be "from files". (for this option
      files with a given pattern will be retrieved  and the ### will
      be used to mark an ID part from the filename.
      - For each file a function to process it will be called
        (_importFile(fileName, fileId))
    """

    IMPORT_FROM_FILES = 0

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form: Form) -> None:
        form.addSection(label="Import")

        form.addParam(
            "filePath",
            params.PathParam,
            label="File path",
            help="Path to the file you want to import",
        )
        form.addParam(
            "copyFiles",
            params.BooleanParam,
            default=False,
            expertLevel=params.LEVEL_ADVANCED,
            label="Copy file?",
            help="By default the file is not copied into the "
            "project to avoid data duplication and to save "
            "disk space. Instead of copying, symbolic links are "
            "created pointing to original files. This approach "
            "has the drawback that if the project is moved to "
            "another computer, the links need to be restored.",
        )

        self._defineImportParams(form)

        self._defineAcquisitionParams(form)

        self._defineBlacklistParams(form)

    def _defineImportParams(self, form: Form) -> None:
        """Override to add options related to the different types
        of import that are allowed by each protocol.
        """
        pass

    def _defineAcquisitionParams(self, form: Form) -> None:
        """Override to add options related to acquisition info."""
        pass

    def _defineBlacklistParams(self, form: Form) -> None:
        """Override to add options related to blacklist info."""
        pass

    def _getDefaultChoice(self) -> int:
        return self.IMPORT_FROM_FILES

    # --------------------------- INFO functions ------------------------------
    def _validate(self) -> List[str]:
        errors = []
        if not os.path.isfile(self.filePath.get()):
            errors.append(f"{self.filePath.get()} is not a file.")
        return errors

    # --------------------------- BASE methods to be overwritten ----------------
    def _getImportChoices(self) -> List[str]:
        """Return a list of possible choices
        from which the import can be done.
        (usually packages formats such as: xmipp3, eman2, relion...etc.
        """
        return ["files"]

    # --------------------------- UTILS functions -----------------------------
    def isBlacklisted(self, fileName: str) -> bool:
        """Overwrite in subclasses"""
        return False
