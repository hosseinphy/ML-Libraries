from gan_manager import GANManager
import sys

def process_cmdline():
    argList = sys.argv
    argc = len(argList)

    # expected: instance_uuid
    if argc-1 != 2:
        raise Exception("run with incorrect number of arguments ({} != 1)".format(argc-1))

    return argList[1:]


if __name__=="__main__":
    
    # voxel_name = sys.argv[1] if sys.argv[1] else "voxel.npy"
    instance_uuid, voxel_name = process_cmdline()
    gm = GANManager()
    # gm.generate_structure(instance_uuid)
    gm.generator_draw_issues(f"instances/{instance_uuid}/{voxel_name}")

