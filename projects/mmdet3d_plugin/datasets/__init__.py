from .nuscenes_dataset import CustomNuScenesDataset
from .builder import custom_build_dataset

from .once.once_dataset import ONCEDataset
from .once.once_temporal_dataset import ONCETemporalDataset
# from .once.once_semi_dataset import ONCEPretrainDataset, ONCELabeledDataset, ONCEUnlabeledDataset, ONCETestDataset, split_once_semi_data

# _semi_dataset_dict = {
#     'ONCEDataset': {
#         'PARTITION_FUNC': split_once_semi_data,
#         'PRETRAIN': ONCEPretrainDataset,
#         'LABELED': ONCELabeledDataset,
#         'UNLABELED': ONCEUnlabeledDataset,
#         'TEST': ONCETestDataset
#     }
# }

from .waymo.waymo_mv_dataset import CustomWaymoDataset
from .waymo.waymo_mv_temporal_dataset import CustomTemporalWaymoDataset

__all__ = [
    'CustomNuScenesDataset','ONCEDataset','CustomWaymoDataset','CustomTemporalWaymoDataset','ONCEPretrainDataset','ONCELabeledDataset','ONCEUnlabeledDataset','ONCETestDataset','ONCETemporalDataset'
]


# try:
#     from .waymo_mv_dataset import CustomWaymoDataset
#     from .waymo_mv_temporal_dataset import CustomTemporalWaymoDataset
#     __all__.append('CustomWaymoDataset')
#     __all__.append('CustomTemporalWaymoDataset')

# except:
#     print('cannot import waymo dataset')