# coding=utf-8


import os

import numpy as np
from pyworkflow import BETA
from pyworkflow.utils.path import removeExt
from scipy.ndimage import center_of_mass, shift

from pwfluo.objects import (
    FluoImage,
    Particle,
    PSFModel,
    SetOfFluoImages,
    SetOfParticles,
)
from pwfluo.protocols.protocol_base import ProtFluoImportFile, ProtFluoImportFiles


def move_center_of_mass_to_center(volume: np.ndarray, order: int = 1):
    tvec = (np.asarray(volume.shape) - 1) / 2 - np.asarray(center_of_mass(volume))
    return shift(volume, np.asarray(tvec), order=order)


class ProtImportFluoImages(ProtFluoImportFiles):
    """Protocol to import a set of fluoimages to the project"""

    OUTPUT_NAME = "FluoImages"

    _outputClassName = "SetOfFluoImages"
    _label = "import fluoimages"
    _devStatus = BETA
    _possibleOutputs = {OUTPUT_NAME: SetOfFluoImages}

    def _insertAllSteps(self):
        self._insertFunctionStep(
            self.importStep,
            FluoImage,
        )

    def _getMessage(self) -> str:
        return "FluoImages %s" % self.getObjectTag(self.OUTPUT_NAME)


class ProtImportSetOfParticles(ProtFluoImportFiles):
    """Protocol to import a set of particles to the project"""

    OUTPUT_NAME = "SetOfParticles"

    _outputClassName = "SetOfParticles"
    _label = "import particles"
    _devStatus = BETA
    _possibleOutputs = {OUTPUT_NAME: SetOfParticles}

    def _insertAllSteps(self):
        self._insertFunctionStep(
            self.importStep,
            Particle,
        )

    # --------------------------- INFO functions ------------------------------
    def _getMessage(self) -> str:
        return "Particles %s" % self.getObjectTag(self.OUTPUT_NAME)


class ProtImportFluoImage(ProtFluoImportFile):
    """Protocol to import a fluo image to the project"""

    OUTPUT_NAME = "FluoImage"

    _outputClassName = "FluoImage"
    _label = "import fluoimage"
    _devStatus = BETA
    _possibleOutputs = {OUTPUT_NAME: FluoImage}

    def _insertAllSteps(self):
        self._insertFunctionStep(
            self.importImageStep,
            FluoImage,
        )

    # --------------------------- INFO functions ------------------------------
    def _getMessage(self) -> str:
        return "FluoImage %s" % self.getObjectTag(self.OUTPUT_NAME)


class ProtImportPSFModel(ProtFluoImportFile):
    """Protocol to import a fluo image to the project"""

    OUTPUT_NAME = "PSFModel"

    _outputClassName = "PSFModel"
    _label = "import psf"
    _devStatus = BETA
    _possibleOutputs = {OUTPUT_NAME: PSFModel}

    def _insertAllSteps(self):
        self._insertFunctionStep(self.importPSFStep)

    def importPSFStep(self) -> None:
        """Copy the file.
        Register other parameters.
        """
        file_path: str = self.filePath.get()
        img = PSFModel.from_filename(file_path)
        voxel_size: tuple[float, float] = self.vs_xy.get(), self.vs_z.get()
        data = img.getData()
        for c in range(data.shape[0]):
            data[c] = move_center_of_mass_to_center(data[c], order=3)

        # Save PSF
        newFileName = os.path.basename(file_path)
        if not newFileName.endswith(".ome.tiff"):
            newFileName, _ = os.path.splitext(newFileName)
            newFileName = newFileName + ".ome.tiff"
        img = PSFModel.from_data(
            data, self._getExtraPath(newFileName), voxel_size=voxel_size
        )
        imgId = removeExt(newFileName)
        img.setObjId(imgId)

        self._defineOutputs(**{self.OUTPUT_NAME: img})

    # --------------------------- INFO functions ------------------------------
    def _getMessage(self) -> str:
        return "PSFModel %s" % self.getObjectTag(self.OUTPUT_NAME)
