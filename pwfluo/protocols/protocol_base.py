# **************************************************************************
# *
# * Authors:     Jean Plumail (jplumail@unistra.fr) [1]
# *
# * [1] ICube, Université de Strasbourg
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************

from __future__ import annotations

import os
import re
from typing import TypeVar

import pyworkflow as pw
import pyworkflow.protocol.params as params
from pyworkflow import utils as pwutils
from pyworkflow.mapper.sqlite_db import SqliteDb
from pyworkflow.object import Set
from pyworkflow.protocol.params import (
    FloatParam,
    Form,
    PointerParam,
)
from pyworkflow.utils.path import createAbsLink, removeExt
from pyworkflow.utils.properties import Message

import pwfluo.objects as pwfluoobj
from pwfluo.constants import MICRON_STR
from pwfluo.protocols.import_ import ProtImport, ProtImportFile, ProtImportFiles


def _getUniqueFileName(pattern, filename, filePaths=None):
    if filePaths is None:
        filePaths = [re.split(r"[$*#?]", pattern)[0]]

    commPath = pwutils.commonPath(filePaths)
    return filename.replace(commPath + "/", "").replace("/", "_")


class ProtFluoBase:
    T = TypeVar("T", bound=Set)
    OUTPUT_PREFIX: str

    def _createSet(self, SetClass: type[T], template, suffix, **kwargs):
        """Create a set and set the filename using the suffix.
        If the file exists, it will be deleted."""
        setFn = self._getPath(template % suffix)
        # Close the connection to the database if
        # it is open before deleting the file
        pw.utils.cleanPath(setFn)

        SqliteDb.closeConnection(setFn)
        setObj = SetClass(filename=setFn, **kwargs)
        return setObj

    def _createSetOfCoordinates3D(
        self, volSet: pwfluoobj.SetOfFluoImages, suffix: str = ""
    ):
        coord3DSet = self._createSet(
            pwfluoobj.SetOfCoordinates3D,
            "coordinates%s.sqlite",
            suffix,
            indexes=[pwfluoobj.Coordinate3D.IMAGE_ID_ATTR],
        )
        coord3DSet.setPrecedents(volSet)
        return coord3DSet

    def _createSetOfFluoImages(self, suffix: str = ""):
        return self._createSet(pwfluoobj.SetOfFluoImages, "fluoimages%s.sqlite", suffix)

    def _createSetOfParticles(self, suffix: str = ""):
        return self._createSet(pwfluoobj.SetOfParticles, "particles%s.sqlite", suffix)

    def _getOutputSuffix(self, cls: type):
        """Get the name to be used for a new output.
        For example: output3DCoordinates7.
        It should take into account previous outputs
        and number with a higher value.
        """
        maxCounter = -1
        for attrName, _ in self.iterOutputAttributes(cls):
            suffix = attrName.replace(self.OUTPUT_PREFIX, "")
            try:
                counter = int(suffix)
            except ValueError:
                counter = 1  # when there is not number assume 1
            maxCounter = max(counter, maxCounter)

        return str(maxCounter + 1) if maxCounter > 0 else ""  # empty if not output


class ProtFluoPicking(ProtImport, ProtFluoBase):
    OUTPUT_PREFIX = "output3DCoordinates"

    """ Base class for Fluo boxing protocols. """

    def _defineParams(self, form: Form) -> None:
        form.addSection(label="Input")
        form.addParam(
            "inputFluoImages",
            PointerParam,
            label="Input Images",
            important=True,
            pointerClass="SetOfFluoImages",
            help="Select the Image to be used during picking.",
        )

    def _summary(self) -> list[str]:
        summary = []
        if self.isFinished() and self.getOutputsSize() >= 1:
            for key, output in self.iterOutputAttributes():
                summary.append("*%s:*\n%s" % (key, output.getSummary()))
        else:
            summary.append(Message.TEXT_NO_OUTPUT_CO)
        return summary


class ProtFluoImportBase(ProtFluoBase):
    def __init__(self):
        self.images = None

    def _defineAcquisitionParams(self, form: Form) -> None:
        """Override to add options related to acquisition info."""
        group = form.addGroup("Image Info")
        group.addParam(
            "importWizard",
            params.LabelParam,
            important=True,
            label="Use the wizard button to import parameters.",
            help="The wizard will try to import image parameters.\n"
            "If not found, required ones should be provided.",
        )
        group.addParam("vs_xy", FloatParam, label=f"XY ({MICRON_STR}/px)")
        group.addParam("vs_z", FloatParam, label=f"Z ({MICRON_STR}/px)")

    # --------------------------- INFO functions ------------------------------
    def _getMessage(self) -> str:
        return ""

    def _hasOutput(self):
        return self.images is not None

    def _methods(self) -> list[str]:
        methods = []
        if self._hasOutput():
            vs_xy, vs_z = self.vs_xy.get(), self.vs_z.get()
            methods.append(
                f"{self._getMessage()} imported with a voxel size "
                f"*{vs_xy:.2f}x{vs_z:.2f}* ({MICRON_STR}/px)"
            )
        return methods


class ProtFluoImportFiles(ProtFluoImportBase, ProtImportFiles):
    T = TypeVar("T", bound=pwfluoobj.FluoImage)

    def __init__(self, **args):
        ProtImportFiles.__init__(self, **args)
        ProtFluoImportBase.__init__(self)

    def importStep(self, obj: type[T]):
        """Copy images matching the filename pattern
        Register other parameters.
        """
        pattern = self.getPattern()
        self.info("Using pattern: '%s'" % pattern)

        if obj is pwfluoobj.FluoImage:
            imgSet = self._createSetOfFluoImages()
        elif obj is pwfluoobj.Particle:
            imgSet = self._createSetOfParticles()
        else:
            raise NotImplementedError()
        voxel_size: tuple[float, float] = self.vs_xy.get(), self.vs_z.get()
        imgSet.setVoxelSize(voxel_size)

        fileNameList = []
        for fileName, fileId in self.iterFiles():
            img = obj.from_filename(fileName)
            img.setVoxelSize(voxel_size)

            newFileName = os.path.basename(fileName)
            if newFileName in fileNameList:
                newFileName = _getUniqueFileName(self.getPattern(), fileName)

            # If not OME-TIFF file, copy file to .ome.tiff
            if not newFileName.endswith(".ome.tiff"):
                newFileName, _ = os.path.splitext(newFileName)
                newFileName = newFileName + ".ome.tiff"
                newFilePath = self._getExtraPath(newFileName)
                img = obj.from_data(img.getData(), newFilePath, voxel_size=voxel_size)
            else:
                newFilePath = self._getExtraPath(newFileName)
                createAbsLink(
                    os.path.abspath(fileName),
                    os.path.abspath(newFilePath),
                )
                img.setFileName(newFilePath)

            fileNameList.append(newFileName)

            imgId = removeExt(newFileName)
            img.setImgId(imgId)
            img.cleanObjId()

            imgSet.append(img)

        imgSet.write()
        self._defineOutputs(**{self.OUTPUT_NAME: imgSet})

    def load_image_info(self):
        """Return a proper acquisitionInfo (dict)
        or an error message (str).
        """
        metadatas = dict(voxel_size=[], image_dim=[], num_channels=[], filename=[])
        for fname, _ in self.iterFiles():
            metadata = pwfluoobj.FluoImage.metadata_from_filename(fname)
            metadatas["filename"].append(fname)
            for k in metadata:
                metadatas[k].append(metadata[k])
        return metadatas

    # --------------------------- INFO functions ------------------------------
    def _summary(self) -> list[str]:
        try:
            summary = []
            if self._hasOutput():
                summary.append(
                    "%s imported from:\n%s" % (self._getMessage(), self.getPattern())
                )

                if (vs_xy := self.vs_xy.get()) and (vs_z := self.vs_z.get()):
                    summary.append(
                        f"Voxel size: *{vs_xy:.2f}x{vs_z:.2f}* ({MICRON_STR}/px)"
                    )

        except Exception as e:
            print(e)

        return summary

    def _getVolumeFileName(self, fileName: str, extension: str | None = None) -> str:
        if extension is not None:
            baseFileName = (
                "import_"
                + str(os.path.basename(fileName)).split(".")[0]
                + ".%s" % extension
            )
        else:
            baseFileName = "import_" + str(os.path.basename(fileName)).split(":")[0]

        return self._getExtraPath(baseFileName)

    def _validate(self) -> list[str]:
        errors = []
        try:
            next(self.iterFiles())
        except StopIteration:
            errors.append(
                "No files matching the pattern %s were found." % self.getPattern()
            )
        return errors


class ProtFluoImportFile(ProtFluoImportBase, ProtImportFile):
    T = TypeVar("T", bound=pwfluoobj.Image)

    def __init__(self, **args):
        ProtFluoImportBase.__init__(self)
        ProtImportFile.__init__(self, **args)

    def importImageStep(self, obj: type[T]) -> None:
        """Copy the file.
        Register other parameters.
        """
        self.info("")

        file_path = self.filePath.get()
        img = obj.from_filename(file_path)
        voxel_size: tuple[float, float] = self.vs_xy.get(), self.vs_z.get()
        img.setVoxelSize(voxel_size)

        # Set default origin
        origin = pwfluoobj.Transform()
        dim = img.getDim()
        if dim is None:
            raise ValueError("Image '%s' has no dimension" % file_path)
        x, y, z = dim
        origin.setShifts(
            x / -2.0 * voxel_size[0],
            y / -2.0 * voxel_size[0],
            z / -2.0 * voxel_size[1],
        )
        img.setOrigin(origin)

        newFileName = os.path.basename(file_path)

        imgId = removeExt(newFileName)
        if isinstance(img, pwfluoobj.FluoImage):
            img.setImgId(imgId)
        else:
            img.setObjId(imgId)

        createAbsLink(
            os.path.abspath(file_path), os.path.abspath(self._getExtraPath(newFileName))
        )

        img.cleanObjId()
        img.setFileName(self._getExtraPath(newFileName))

        self._defineOutputs(**{self.OUTPUT_NAME: img})

    def load_image_info(self):
        """Return a proper acquisitionInfo (dict)
        or an error message (str).
        """
        fname = self.filePath.get()
        metadata = pwfluoobj.FluoImage.metadata_from_filename(fname)
        return dict(
            voxel_size=[metadata["voxel_size"]],
            image_dim=[metadata["image_dim"]],
            num_channels=[metadata["num_channels"]],
            filename=[fname],
        )

    # --------------------------- INFO functions ------------------------------
    def _summary(self) -> list[str]:
        try:
            summary = []
            if self._hasOutput():
                summary.append(
                    "%s imported from:\n%s" % (self._getMessage(), self.getPattern())
                )

                if (vs_xy := self.vs_xy.get()) and (vs_z := self.vs_z.get()):
                    summary.append(
                        f"Voxel size: *{vs_xy:.2f}x{vs_z:.2f}* ({MICRON_STR}/px)"
                    )

        except Exception as e:
            print(e)

        return summary

    def _getVolumeFileName(self, fileName: str, extension: str | None = None) -> str:
        if extension is not None:
            baseFileName = (
                "import_"
                + str(os.path.basename(fileName)).split(".")[0]
                + ".%s" % extension
            )
        else:
            baseFileName = "import_" + str(os.path.basename(fileName)).split(":")[0]

        return self._getExtraPath(baseFileName)
