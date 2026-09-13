import model.modelfactory as mf
from model.meta_learner import (
    AEMetaLearner,
    MetaLearingClassification,
    NeuromodulationMetaLearner,
    OMLMetaLearner,
)
from model.meta_learner_fseml import MetaLearnerFSEML


class MetaLearnerFactory:
    @staticmethod
    def build(args):
        treatment = args.treatment

        if treatment == "FSEML":
            return MetaLearnerFSEML(args)

        config = mf.ModelFactory.get_model(treatment, args.dataset)
        builders = {
            "AE": lambda: AEMetaLearner(args, config),
            "OML": lambda: OMLMetaLearner(args, config),
            "Neuromodulation": lambda: NeuromodulationMetaLearner(args, config),
        }
        if treatment in builders:
            return builders[treatment]()

        return MetaLearingClassification(args, config, treatment)
