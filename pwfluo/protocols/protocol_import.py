# coding=utf-8

import os
import re
from os.path import abspath, basename
from typing import List, Optional, Tuple

import pyworkflow.protocol.params as params
from aicsimageio import AICSImage
from pyworkflow import BETA
from pyworkflow import utils as pwutils
from pyworkflow.protocol.params import Form
from pyworkflow.utils.path import createAbsLink, removeExt

from pwfluo.objects import (
    FluoImage,
    Particle,
    PSFModel,
    SetOfFluoImages,
    SetOfParticles,
    Transform,
)
from pwfluo.protocols.protocol_base import ProtFluoImportFile, ProtFluoImportFiles


def _getUniqueFileName(pattern, filename, filePaths=None):
    if filePaths is None:
        filePaths = [re.split(r"[$*#?]", pattern)[0]]

    commPath = pwutils.commonPath(filePaths)
    return filename.replace(commPath + "/", "").replace("/", "_")


class ProtImportFluoImages(ProtFluoImportFiles):
    """Protocol to import a set of fluoimages to the project"""

    OUTPUT_NAME = "FluoImages"

    _outputClassName = "SetOfFluoImages"
    _label = "import fluoimages"
    _devStatus = BETA
    _possibleOutputs = {OUTPUT_NAME: SetOfFluoImages}
    READERS = list(map(lambda x: getattr(x, "__name__"), AICSImage.SUPPORTED_READERS))

    def __init__(self, **args):
        ProtFluoImportFiles.__init__(self, **args)
        self.FluoImages: Optional[SetOfFluoImages] = None

    def _defineImportParams(self, form: Form) -> None:
        form.addParam(
            "reader",
            params.EnumParam,
            choices=self.READERS,
            display=params.EnumParam.DISPLAY_COMBO,
            label="Reader",
            help="Choose the reader"
            "The DefaultReader finds a reader that works for your image."
            "BioformatsReader corresponds to the ImageJ reader"
            "(requires java and maven to be installed)",
        )

    def _getImportChoices(self):
        """Return a list of possible choices
        from which the import can be done.
        """
        return ["eman"]

    def _insertAllSteps(self):
        self._insertFunctionStep(
            "importFluoImagesStep",
            self.getPattern(),
            (self.vs_xy.get(), self.vs_z.get()),
        )

    # --------------------------- STEPS functions -----------------------------

    def importFluoImagesStep(
        self, pattern: str, voxelSize: Tuple[float, float]
    ) -> None:
        """Copy images matching the filename pattern
        Register other parameters.
        """
        self.info("Using pattern: '%s'" % pattern)

        imgSet = self._createSetOfFluoImages()
        imgSet.setVoxelSize(voxelSize)

        fileNameList = []
        for fileName, fileId in self.iterFiles():
            img = FluoImage(
                data=fileName, reader=AICSImage.SUPPORTED_READERS[self.reader.get()]
            )
            img.setVoxelSize(voxelSize)

            # Set default origin
            origin = Transform()
            dim = img.getDim()
            if dim is None:
                raise ValueError("Image '%s' has no dimension" % fileName)
            x, y, z = dim
            origin.setShifts(
                x / -2.0 * voxelSize[0],
                y / -2.0 * voxelSize[0],
                z / -2.0 * voxelSize[1],
            )
            img.setOrigin(origin)

            newFileName = basename(fileName).split(":")[0]
            if newFileName in fileNameList:
                newFileName = _getUniqueFileName(
                    self.getPattern(), fileName.split(":")[0]
                )

            fileNameList.append(newFileName)

            imgId = removeExt(newFileName)
            img.setImgId(imgId)

            createAbsLink(abspath(fileName), abspath(self._getExtraPath(newFileName)))

            img.cleanObjId()
            img.setFileName(self._getExtraPath(newFileName))
            imgSet.append(img)

        imgSet.write()
        self._defineOutputs(**{self.OUTPUT_NAME: imgSet})

    # --------------------------- INFO functions ------------------------------
    def _hasOutput(self) -> bool:
        return self.FluoImages is not None

    def _getTomMessage(self) -> str:
        return "FluoImages %s" % self.getObjectTag(self.OUTPUT_NAME)

    def _summary(self) -> List[str]:
        try:
            summary = []
            if self._hasOutput():
                summary.append(
                    "%s imported from:\n%s" % (self._getTomMessage(), self.getPattern())
                )

                if (vs_xy := self.vs_xy.get()) and (vs_z := self.vs_z.get()):
                    summary.append(f"Voxel size: *{vs_xy:.2f}x{vs_z:.2f}* (μm/px)")

        except Exception as e:
            print(e)

        return summary

    def _methods(self) -> List[str]:
        methods = []
        if self._hasOutput():
            vs_xy, vs_z = self.vs_xy.get(), self.vs_z.get()
            methods.append(
                f"{self._getTomMessage()} imported with a voxel size "
                f"*{vs_xy:.2f}x{vs_z:.2f}* (μm/px)"
            )
        return methods

    def _getVolumeFileName(self, fileName: str, extension: Optional[str] = None) -> str:
        if extension is not None:
            baseFileName = (
                "import_" + str(basename(fileName)).split(".")[0] + ".%s" % extension
            )
        else:
            baseFileName = "import_" + str(basename(fileName)).split(":")[0]

        return self._getExtraPath(baseFileName)

    def _validate(self) -> List[str]:
        errors = []
        try:
            next(self.iterFiles())
        except StopIteration:
            errors.append(
                "No files matching the pattern %s were found." % self.getPattern()
            )
        return errors


# TODO: refactor classes
class ProtImportSetOfParticles(ProtFluoImportFiles):
    """Protocol to import a set of particles to the project"""

    OUTPUT_NAME = "SetOfParticles"

    _outputClassName = "SetOfParticles"
    _label = "import particles"
    _devStatus = BETA
    _possibleOutputs = {OUTPUT_NAME: SetOfParticles}

    def __init__(self, **args):
        ProtFluoImportFiles.__init__(self, **args)
        self.Particles: Optional[SetOfParticles] = None

    def _getImportChoices(self):
        """Return a list of possible choices
        from which the import can be done.
        """
        return ["eman"]

    def _insertAllSteps(self):
        self._insertFunctionStep(
            self.importParticlesStep,
            self.getPattern(),
            (self.vs_xy.get(), self.vs_z.get()),
        )

    # --------------------------- STEPS functions -----------------------------

    def importParticlesStep(self, pattern: str, voxelSize: Tuple[float, float]) -> None:
        """Copy images matching the filename pattern
        Register other parameters.
        """
        self.info("Using pattern: '%s'" % pattern)

        particles = self._createSetOfParticles()
        particles.setVoxelSize(voxelSize)

        fileNameList = []
        for fileName, fileId in self.iterFiles():
            particle = Particle(data=fileName)
            particle.setVoxelSize(voxelSize)

            # Set default origin
            origin = Transform()
            dim = particle.getDim()
            if dim is None:
                raise ValueError("Image '%s' has no dimension" % fileName)
            x, y, z = dim
            origin.setShifts(
                x / -2.0 * voxelSize[0],
                y / -2.0 * voxelSize[0],
                z / -2.0 * voxelSize[1],
            )
            particle.setOrigin(origin)

            newFileName = basename(fileName).split(":")[0]
            if newFileName in fileNameList:
                newFileName = _getUniqueFileName(
                    self.getPattern(), fileName.split(":")[0]
                )

            fileNameList.append(newFileName)

            imgId = removeExt(newFileName)
            particle.setImgId(imgId)

            createAbsLink(abspath(fileName), abspath(self._getExtraPath(newFileName)))

            particle.cleanObjId()
            particle.setFileName(self._getExtraPath(newFileName))
            particles.append(particle)

        particles.write()
        self._defineOutputs(**{self.OUTPUT_NAME: particles})

    # --------------------------- INFO functions ------------------------------
    def _hasOutput(self) -> bool:
        return self.Particles is not None

    def _getTomMessage(self) -> str:
        return "Particles %s" % self.getObjectTag(self.OUTPUT_NAME)

    def _summary(self) -> List[str]:
        try:
            summary = []
            if self._hasOutput():
                summary.append(
                    "%s imported from:\n%s" % (self._getTomMessage(), self.getPattern())
                )

                if (vs_xy := self.vs_xy.get()) and (vs_z := self.vs_z.get()):
                    summary.append(f"Voxel size: *{vs_xy:.2f}x{vs_z:.2f}* (μm/px)")

        except Exception as e:
            print(e)

        return summary

    def _methods(self) -> List[str]:
        methods = []
        if self._hasOutput():
            vs_xy, vs_z = self.vs_xy.get(), self.vs_z.get()
            methods.append(
                f"{self._getTomMessage()} imported with a voxel size "
                f"*{vs_xy:.2f}x{vs_z:.2f}* (μm/px)"
            )
        return methods

    def _getVolumeFileName(self, fileName: str, extension: Optional[str] = None) -> str:
        if extension is not None:
            baseFileName = (
                "import_" + str(basename(fileName)).split(".")[0] + ".%s" % extension
            )
        else:
            baseFileName = "import_" + str(basename(fileName)).split(":")[0]

        return self._getExtraPath(baseFileName)

    def _validate(self) -> List[str]:
        errors = []
        try:
            next(self.iterFiles())
        except StopIteration:
            errors.append(
                "No files matching the pattern %s were found." % self.getPattern()
            )
        return errors


# TODO: refactor classes
class ProtImportFluoImage(ProtFluoImportFile):
    """Protocol to import a fluo image to the project"""

    OUTPUT_NAME = "FluoImage"

    _outputClassName = "FluoImage"
    _label = "import fluoimage"
    _devStatus = BETA
    _possibleOutputs = {OUTPUT_NAME: FluoImage}
    READERS = list(map(lambda x: getattr(x, "__name__"), AICSImage.SUPPORTED_READERS))

    def __init__(self, **args):
        ProtFluoImportFile.__init__(self, **args)
        self.FluoImage: Optional[FluoImage] = None

    def _defineParams(self, form):
        ProtFluoImportFile._defineParams(self, form)

    def _defineImportParams(self, form: Form) -> None:
        form.addParam(
            "reader",
            params.EnumParam,
            choices=self.READERS,
            label="Reader",
            help="Choose the reader"
            "The DefaultReader finds a reader that works for your image."
            "BioformatsReader corresponds to the ImageJ reader"
            "(requires java and maven to be installed)",
        )

    def _getImportChoices(self):  # TODO: remove this
        """Return a list of possible choices
        from which the import can be done.
        """
        return ["eman"]

    def _insertAllSteps(self):
        self._insertFunctionStep(
            "importFluoImageStep",
            self.filePath.get(),
            (self.vs_xy.get(), self.vs_z.get()),
        )

    # --------------------------- STEPS functions -----------------------------

    def importFluoImageStep(
        self, file_path: str, voxelSize: Tuple[float, float]
    ) -> None:
        """Copy the file.
        Register other parameters.
        """
        self.info("")

        img = FluoImage(
            data=file_path, reader=AICSImage.SUPPORTED_READERS[self.reader.get()]
        )
        img.setVoxelSize(voxelSize)

        # Set default origin
        origin = Transform()
        dim = img.getDim()
        if dim is None:
            raise ValueError("Image '%s' has no dimension" % file_path)
        x, y, z = dim
        origin.setShifts(
            x / -2.0 * voxelSize[0],
            y / -2.0 * voxelSize[0],
            z / -2.0 * voxelSize[1],
        )
        img.setOrigin(origin)

        newFileName = basename(file_path)

        imgId = removeExt(newFileName)
        img.setImgId(imgId)

        createAbsLink(abspath(file_path), abspath(self._getExtraPath(newFileName)))

        img.cleanObjId()
        img.setFileName(self._getExtraPath(newFileName))

        self._defineOutputs(**{self.OUTPUT_NAME: img})

    # --------------------------- INFO functions ------------------------------
    def _hasOutput(self) -> bool:
        return self.FluoImage is not None

    def _getTomMessage(self) -> str:
        return "FluoImage %s" % self.getObjectTag(self.OUTPUT_NAME)

    def _summary(self) -> List[str]:
        try:
            summary = []
            if self._hasOutput():
                summary.append(
                    "%s imported from:\n%s"
                    % (self._getTomMessage(), self.filePath.get())
                )

                if (vs_xy := self.vs_xy.get()) and (vs_z := self.vs_z.get()):
                    summary.append(f"Voxel size: *{vs_xy:.2f}x{vs_z:.2f}* (μm/px)")

        except Exception as e:
            print(e)

        return summary

    def _methods(self) -> List[str]:
        methods = []
        if self._hasOutput():
            vs_xy, vs_z = self.vs_xy.get(), self.vs_z.get()
            methods.append(
                f"{self._getTomMessage()} imported with a voxel size "
                f"*{vs_xy:.2f}x{vs_z:.2f}* (μm/px)"
            )
        return methods

    def _getVolumeFileName(self, fileName: str, extension: Optional[str] = None) -> str:
        if extension is not None:
            baseFileName = (
                "import_" + str(basename(fileName)).split(".")[0] + ".%s" % extension
            )
        else:
            baseFileName = "import_" + str(basename(fileName)).split(":")[0]

        return self._getExtraPath(baseFileName)

    def _validate(self) -> List[str]:
        errors = []
        if not os.path.isfile(self.filePath.get()):
            errors.append(f"{self.filePath.get()} is not a file.")
        return errors


class ProtImportPSFModel(ProtFluoImportFile):
    """Protocol to import a psf to the project"""

    OUTPUT_NAME = "PSFModel"

    _outputClassName = "PSFModel"
    _label = "import psf"
    _devStatus = BETA
    _possibleOutputs = {OUTPUT_NAME: PSFModel}
    READERS = list(map(lambda x: getattr(x, "__name__"), AICSImage.SUPPORTED_READERS))

    def __init__(self, **args):
        ProtFluoImportFile.__init__(self, **args)
        self.PSFModel: Optional[PSFModel] = None

    def _defineParams(self, form):
        ProtFluoImportFile._defineParams(self, form)

    def _defineImportParams(self, form: Form) -> None:
        form.addParam(
            "reader",
            params.EnumParam,
            choices=self.READERS,
            display=params.EnumParam.DISPLAY_COMBO,
            label="Reader",
            help="Choose the reader"
            "The DefaultReader finds a reader that works for your image."
            "BioformatsReader corresponds to the ImageJ reader"
            "(requires java and maven to be installed)",
        )

    def _getImportChoices(self):  # TODO: remove this
        """Return a list of possible choices
        from which the import can be done.
        """
        return ["eman"]

    def _insertAllSteps(self):
        self._insertFunctionStep(
            "importPSFModelStep",
            self.filePath.get(),
            (self.vs_xy.get(), self.vs_z.get()),
        )

    # --------------------------- STEPS functions -----------------------------

    def importPSFModelStep(
        self, file_path: str, voxelSize: Tuple[float, float]
    ) -> None:
        """Copy the file.
        Register other parameters.
        """
        self.info("")

        img = PSFModel(
            data=file_path, reader=AICSImage.SUPPORTED_READERS[self.reader.get()]
        )
        img.setVoxelSize(voxelSize)

        # Set default origin
        origin = Transform()
        dim = img.getDim()
        if dim is None:
            raise ValueError("Image '%s' has no dimension" % file_path)
        x, y, z = dim
        origin.setShifts(
            x / -2.0 * voxelSize[0],
            y / -2.0 * voxelSize[0],
            z / -2.0 * voxelSize[1],
        )
        img.setOrigin(origin)

        newFileName = basename(file_path)

        createAbsLink(abspath(file_path), abspath(self._getExtraPath(newFileName)))

        img.cleanObjId()
        img.setFileName(self._getExtraPath(newFileName))

        self._defineOutputs(**{self.OUTPUT_NAME: img})

    # --------------------------- INFO functions ------------------------------
    def _hasOutput(self) -> bool:
        return self.PSFModel is not None

    def _getTomMessage(self) -> str:
        return "PSFModel %s" % self.getObjectTag(self.OUTPUT_NAME)

    def _summary(self) -> List[str]:
        try:
            summary = []
            if self._hasOutput():
                summary.append(
                    "%s imported from:\n%s"
                    % (self._getTomMessage(), self.filePath.get())
                )

                if (vs_xy := self.vs_xy.get()) and (vs_z := self.vs_z.get()):
                    summary.append(f"Voxel size: *{vs_xy:.2f}x{vs_z:.2f}* (μm/px)")

        except Exception as e:
            print(e)

        return summary

    def _methods(self) -> List[str]:
        methods = []
        if self._hasOutput():
            vs_xy, vs_z = self.vs_xy.get(), self.vs_z.get()
            methods.append(
                f"{self._getTomMessage()} imported with a voxel size "
                f"*{vs_xy:.2f}x{vs_z:.2f}* (μm/px)"
            )
        return methods

    def _getVolumeFileName(self, fileName: str, extension: Optional[str] = None) -> str:
        if extension is not None:
            baseFileName = (
                "import_" + str(basename(fileName)).split(".")[0] + ".%s" % extension
            )
        else:
            baseFileName = "import_" + str(basename(fileName)).split(":")[0]

        return self._getExtraPath(baseFileName)

    def _validate(self) -> List[str]:
        errors = []
        if not os.path.isfile(self.filePath.get()):
            errors.append(f"{self.filePath.get()} is not a file.")
        return errors
