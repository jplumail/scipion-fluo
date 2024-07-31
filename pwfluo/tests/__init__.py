from pyworkflow.tests import DataSet

DataSet(
    name="fluo",
    folder="fluo-data",
    files={
        "isotropic-particles-dir": "generated/isotropic-1.0/particles",
        "isotropic-psf": "generated/isotropic-1.0/psf.tiff",
    },
)
