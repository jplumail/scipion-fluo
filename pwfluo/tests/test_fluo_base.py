from pyworkflow.tests import BaseTest, DataSet, setupTestProject

from pwfluo.protocols import ProtImportPSFModel, ProtImportSetOfParticles


class TestProtocolFluoBase(BaseTest):
    ds = None

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.ds: DataSet = DataSet.getDataSet("fluo")
        cls.runImports()

    @classmethod
    def runImports(cls):
        cls.particles = cls.runImportSetOfParticles()
        cls.psf = cls.runImportPSF()

    @classmethod
    def runImportSetOfParticles(cls):
        prot = cls.newProtocol(
            ProtImportSetOfParticles,
            filesPath=cls.ds.getFile("isotropic-particles-dir") + "/*",
            vs_xy=1.0,
            vs_z=1.0,
        )

        cls.launchProtocol(prot)
        output = prot.SetOfParticles
        return output

    @classmethod
    def runImportPSF(cls):
        protImportPSFModel = cls.newProtocol(
            ProtImportPSFModel,
            filePath=cls.ds.getFile("isotropic-psf"),
            vs_xy=1.0,
            vs_z=1.0,
        )

        cls.launchProtocol(protImportPSFModel)
        psfImported = protImportPSFModel.PSFModel
        return psfImported

    def test_import_data(self):
        self.assertIsNotNone(self.particles)
        self.assertIsNotNone(self.psf)
