import os   , sys
dir = '/'.join(__file__.split('\\')[:-1])
if not dir in sys.path:
    sys.path.append(dir)
    print(dir)

from GSO_image import * 

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir",type=str,required=True,help="Path GSO dataset",)
parser.add_argument("--output_dir", type=str, default="")
parser.add_argument("--num_images", type=int, default=8)


argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)


if args.output_dir == "":
    list_elem = args.input_dir.split('/')
    print(list_elem)
    if "" in list_elem:
        list_elem.pop("")
    args.output_dir = "/".join(list_elem[:-1])+'/GSO_image'
    


obj_pathes = os.listdir(args.input_dir+'/')
for obj_path in obj_pathes:
    if not os.path.exists(args.output_dir+'/'+obj_path):
        os.makedirs(args.output_dir+'/'+obj_path)
    reset_scene()
    set_light()
    set_camera()
    print(obj_path)
    metadata = {}
    metadata['cam_world_matrix'] = np.zeros((args.num_images,4,4))
    metadata['cam_lens'] = bpy.data.objects["Camera"].data.lens
    try:
        import_obj(args.input_dir + '/' + obj_path)
        for i in range(args.num_images):
            metadata['cam_world_matrix'][i] = bpy.data.objects["Camera"].matrix_world
            randomize_camera()
            bpy.context.scene.render.filepath = args.output_dir+'/'+obj_path+'/'+str(i)+'.png'
            bpy.ops.render.render(write_still=True)
    except Exception as e:
        print("Failed to render", obj_path)
    metadata['cam_world_matrix'] = metadata['cam_world_matrix'].tolist()
    with open(args.output_dir+'/'+obj_path+"/metadata.json", "w") as outfile: 
        json.dump(metadata, outfile)