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

from typing import List, Type, TypeVar

import pyworkflow as pw
from pyworkflow.mapper.sqlite_db import SqliteDb
from pyworkflow.object import Set
from pyworkflow.protocol import Protocol
from pyworkflow.protocol.params import (
    FloatParam,
    Form,
    PointerParam,
)
from pyworkflow.utils.properties import Message

import pwfluo.objects as pwfluoobj
from pwfluo.protocols.import_ import ProtImport, ProtImportFile, ProtImportFiles


class ProtFluoBase:
    T = TypeVar("T", bound=Set)
    OUTPUT_PREFIX: str

    def _createSet(self, SetClass: Type[T], template, suffix, **kwargs) -> T:
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
    ) -> pwfluoobj.SetOfCoordinates3D:
        coord3DSet: pwfluoobj.SetOfCoordinates3D = self._createSet(
            pwfluoobj.SetOfCoordinates3D,
            "coordinates%s.sqlite",
            suffix,
            indexes=[pwfluoobj.Coordinate3D.IMAGE_ID_ATTR],
        )
        coord3DSet.setPrecedents(volSet)
        return coord3DSet

    def _createSetOfFluoImages(self, suffix: str = "") -> pwfluoobj.SetOfFluoImages:
        return self._createSet(pwfluoobj.SetOfFluoImages, "fluoimages%s.sqlite", suffix)

    def _createSetOfParticles(self, suffix: str = "") -> pwfluoobj.SetOfParticles:
        return self._createSet(pwfluoobj.SetOfParticles, "particles%s.sqlite", suffix)

    def _getOutputSuffix(self, cls: type) -> str:
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

    def _summary(self) -> List[str]:
        summary = []
        if self.isFinished() and self.getOutputsSize() >= 1:
            for key, output in self.iterOutputAttributes():
                summary.append("*%s:*\n%s" % (key, output.getSummary()))
        else:
            summary.append(Message.TEXT_NO_OUTPUT_CO)
        return summary


class ProtFluoImportFiles(ProtImportFiles, ProtFluoBase):
    def _defineAcquisitionParams(self, form: Form) -> None:
        """Override to add options related to acquisition info."""
        form.addGroup("Voxel size")
        form.addParam("vs_xy", FloatParam, label="XY (μm/px)")
        form.addParam("vs_z", FloatParam, label="Z (μm/px)")

    def _validate(self):
        pass


class ProtFluoImportFile(
    ProtImportFile, ProtFluoBase
):  # TODO: find a better architecture
    def _defineAcquisitionParams(self, form: Form) -> None:
        """Override to add options related to acquisition info."""
        form.addGroup("Voxel size")
        form.addParam("vs_xy", FloatParam, label="XY (μm/px)")
        form.addParam("vs_z", FloatParam, label="Z (μm/px)")

    def _validate(self):
        pass


class ProtFluoParticleAveraging(Protocol, ProtFluoBase):
    """Base class for subtomogram averaging protocols."""

    pass
