"""
Project configurations.
"""
import os


class Config(object):
    """ Datasets and experiments folders for the whole project. """
    #####################
    ## Dataset setting ##
    #####################
    default_dataroot = os.path.join(
        os.path.dirname(__file__),
        '..','..','..','data-ssl'
    )
    default_dataroot = os.path.abspath(default_dataroot)
    default_exproot = os.path.join(
        os.path.dirname(__file__),
        '..','..','..','exp-ssl'
    )
    default_exproot = os.path.abspath(default_exproot)

    DATASET_ROOT = os.getenv("DATASET_ROOT", default_dataroot)  # TODO: path to your datasets folder
    if not os.path.exists(DATASET_ROOT):
        os.makedirs(DATASET_ROOT, exist_ok=True)
    
    # Synthetic shape dataset
    synthetic_dataroot = os.path.join(DATASET_ROOT, "synthetic_shapes")
    synthetic_cache_path = os.path.join(DATASET_ROOT, "synthetic_shapes")
    if not os.path.exists(synthetic_dataroot):
        os.makedirs(synthetic_dataroot, exist_ok=True)
    
    EXPORT_ROOT = os.getenv("EXPORT_ROOT", default_dataroot)  # TODO: path to your datasets folder
    
    # Exported predictions dataset
    export_dataroot = os.path.join(EXPORT_ROOT, "export_datasets")
    export_cache_path = os.path.join(EXPORT_ROOT, "export_datasets")
    if not os.path.exists(export_dataroot):
        os.makedirs(export_dataroot, exist_ok=True)
    
    # York Urban dataset
    yorkurban_dataroot = os.path.join(DATASET_ROOT, "YorkUrbanDB")
    yorkurban_cache_path = os.path.join(DATASET_ROOT, "YorkUrbanDB")

    # Wireframe dataset
    wireframe_dataroot = os.path.join(DATASET_ROOT, "wireframe")
    wireframe_cache_path = os.path.join(DATASET_ROOT, "wireframe")

    # Holicity dataset
    holicity_dataroot = os.path.join(DATASET_ROOT, "Holicity")
    holicity_cache_path = os.path.join(DATASET_ROOT, "Holicity")
    
    # Official York Urban dataset
    official_yorkurban_dataroot = os.path.join(DATASET_ROOT, "off_YorkUrbanDB")
    official_yorkurban_cache_path = os.path.join(DATASET_ROOT, "off_YorkUrbanDB")

    # NYU_depth_v2
    nyu_dataroot = os.path.join(DATASET_ROOT, "NYU_depth_v2")
    nyu_dataroot_cache_path = os.path.join(DATASET_ROOT, "NYU_depth_v2")

    rdnim_dataroot = os.path.join(DATASET_ROOT, "RDNIM")
    hpatches_dataroot = os.path.join(DATASET_ROOT, "HPatches_sequences")

    ########################
    ## Experiment Setting ##
    ########################
    EXP_PATH = os.getenv("EXP_PATH", default_exproot)  # TODO: path to your experiments folder

    if not os.path.exists(EXP_PATH):
        os.makedirs(EXP_PATH, exist_ok=True)
