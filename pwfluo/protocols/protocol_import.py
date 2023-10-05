# coding=utf-8


from pyworkflow import BETA

from pwfluo.objects import (
    FluoImage,
    Particle,
    PSFModel,
    SetOfFluoImages,
    SetOfParticles,
)
from pwfluo.protocols.protocol_base import ProtFluoImportFile, ProtFluoImportFiles


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
        self._insertFunctionStep(
            self.importImageStep,
            PSFModel,
        )

    # --------------------------- INFO functions ------------------------------
    def _getMessage(self) -> str:
        return "PSFModel %s" % self.getObjectTag(self.OUTPUT_NAME)
