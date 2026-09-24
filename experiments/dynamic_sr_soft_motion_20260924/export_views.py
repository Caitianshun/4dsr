#!/usr/bin/env python3
"""CPU fixed native crops and a60-frame comparison video from saved evaluations.

First run --prepare-rois-only, before inspecting predictions. This opens only
HR reference images and freezes image identity/coordinates in roi_protocol.json.
Later use --roi-protocol with --methods to export registered stills and video.
No model loading, rendering, training, GPU scheduling or teacher generation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import sys
import traceback

from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'experiments/dynamic_sr_20260918'
sys.path.insert(0,str(OLD))


def module_from(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


shared=module_from('soft_views_summary_helpers',Path(__file__).with_name('summarize.py'))
sha,read,write,require=shared.sha,shared.read,shared.write,shared.require

# Chosen from only reference cam00/cam02 frame40 on2026-09-24 before reading
# this batch's predictions. Sizes are native168x168; cameras use own boxes.
PRESETS={
 'cook_spinach':{
  'cam00':{'face_reference':[680,424,848,592], 'clothing_reference':[608,568,776,736],
           'hand_utensil_pan_reference':[580,792,748,960], 'background_books_reference':[1136,528,1304,696]},
  'cam02':{'face_reference':[840,424,1008,592], 'clothing_reference':[792,576,960,744],
           'hand_utensil_pan_reference':[680,776,848,944], 'background_appliance_reference':[248,480,416,648]}},
 'cut_roasted_beef':{
  'cam00':{'face_reference':[696,488,864,656], 'clothing_reference':[624,648,792,816],
           'hand_knife_food_reference':[688,792,856,960], 'background_books_reference':[1136,528,1304,696]},
  'cam02':{'face_reference':[792,496,960,664], 'clothing_reference':[784,632,952,800],
           'hand_knife_food_reference':[816,816,984,984], 'background_appliance_reference':[248,480,416,648]}},
 'meetroom_discussion':{
  'cam00':{'face_reference':[568,360,736,528], 'clothing_reference':[552,452,720,620],
           'chair_hand_occlusion_reference':[456,480,624,648], 'background_window_reference':[1092,504,1260,672]},
  'cam02':{'face_reference':[552,284,720,452], 'clothing_reference':[528,368,696,536],
           'chair_hand_occlusion_reference':[448,392,616,560], 'background_window_reference':[1100,448,1268,616]}}
}


def observation(manifest,camera,frame):
    matches=[o for o in manifest['observations'] if (o['camera_id'],o['frame_index'])==(camera,frame)]
    require(len(matches)==1,f'Missing/duplicate observation {camera}/{frame}')
    return matches[0]


def checked_image(path,expected=None):
    digest=sha(path)
    if expected is not None:
        require(digest==expected,f'Image changed: {path}')
    with Image.open(path) as image:
        return image.convert('RGB'),digest


def prepare(args):
    manifest=read(args.manifest)
    spec=read(args.roi_spec) if args.roi_spec else PRESETS.get(manifest['scene'])
    require(spec is not None,'No HR-inspected preset for this scene; supply a fixed --roi-spec before predictions')
    require(set(spec)=={'cam00','cam02'},'This bounded export uses only test cam00 and train cam02')
    width,height=manifest['resolutions']['hr']
    args.out.mkdir(parents=True,exist_ok=True)
    require(not (args.out/'roi_protocol.json').exists(),'Do not overwrite a frozen ROI protocol')
    references=[]
    for camera,boxes in spec.items():
        obs=observation(manifest,camera,40)
        require(obs['split']==('train' if camera=='cam02' else 'test'),'Unexpected fixed visualization split')
        path=args.manifest.parent/obs['hr_path']
        image,digest=checked_image(path,obs['hr_sha256'])
        require(image.size==(width,height),'HR reference size mismatch')
        preview=image.copy();draw=ImageDraw.Draw(preview)
        for name,box in boxes.items():
            require(len(box)==4 and all(isinstance(v,int) for v in box),f'Invalid box{name}')
            x0,y0,x1,y1=box
            require(0<=x0<x1<=width and 0<=y0<y1<=height,f'Box outside image: {box}')
            require(x1-x0==168 and y1-y0==168,'All registered crops are native168x168')
            draw.rectangle((x0,y0,x1-1,y1-1),outline='red',width=2)
            draw.text((x0+3,max(0,y0-14)),name,fill='red',stroke_width=1,stroke_fill='white')
        preview.save(args.out/f'hr_only_roi_registration_{camera}_0040.png')
        references.append({'camera':camera,'frame':40,'path':str(path),'sha256':digest})
    protocol={'status':'fixed_before_prediction_reads','scene':manifest['scene'],
              'manifest':str(args.manifest),'manifest_sha256':sha(args.manifest),
              'reference_images':references,'regions_by_camera_xyxy_exclusive':spec,
              'fixed_still_frames':[40,80],'video_camera':'cam00','video_frames':list(range(0,120,2)),
              'playback_fps':15,'selection':'Chosen from reference HR only, before this batch prediction inspection; unchanged for all methods and both displayed times.',
              'interpretation':'Reference labels describe frame40 content, not semantic masks or tracked regions. Different cameras use separately registered coordinates, not exact physical correspondence.',
              'source_sha256':sha(__file__),'created_utc':datetime.now(timezone.utc).isoformat()}
    write(args.out/'roi_protocol.json',protocol)
    print(json.dumps({'roi_protocol':str(args.out/'roi_protocol.json'),'prediction_files_read':0,'gpu_used':False}))


def comparison_panel(images,title,tile_width=448):
    width,height=next(iter(images.values())).size
    tile_height=round(height*tile_width/width)
    columns=min(3,len(images));rows=math.ceil(len(images)/columns)
    panel=Image.new('RGB',(columns*tile_width,32+rows*(tile_height+24)),'white')
    draw=ImageDraw.Draw(panel);draw.text((5,6),title,fill='black')
    for index,(name,image) in enumerate(images.items()):
        x=(index%columns)*tile_width;y=32+(index//columns)*(tile_height+24)
        draw.text((x+5,y+3),name,fill='black')
        panel.paste(image.resize((tile_width,tile_height),Image.Resampling.LANCZOS),(x,y+24))
    return panel


def export(args):
    require(args.methods is not None and args.roi_protocol is not None,'Export requires --methods and prior --roi-protocol')
    manifest=read(args.manifest);methods=shared.load_methods(args.methods);protocol=read(args.roi_protocol)
    require(protocol['status']=='fixed_before_prediction_reads' and protocol['manifest_sha256']==sha(args.manifest),
            'ROI protocol/manifest mismatch')
    for reference in protocol['reference_images']:
        require(sha(reference['path'])==reference['sha256'],'ROI registration HR changed')
    args.out.mkdir(parents=True,exist_ok=True)
    require(not (args.out/'complete.json').exists(),'Refusing to overwrite completed export')
    evaluations,teacher_config={},None
    for name,item in methods.items():
        config=read(item['train_dir']/'config.json')
        require(config['manifest_sha256']==protocol['manifest_sha256'],f'{name}: wrong manifest')
        if teacher_config is None:
            teacher_config=config
        else:
            require(shared.teacher_identity(config)==shared.teacher_identity(teacher_config),'Teacher identity differs between displayed methods')
        for endpoint in ['train_fixed_6000','test_6000']:
            require(endpoint in item['evaluations'],f'{name}: missing {endpoint}')
            path=item['evaluations'][endpoint]
            path=path if path.name=='metrics.json' else path/'metrics.json'
            values,receipt=read(path),read(path.parent/'complete.json')
            require(receipt['status']=='completed_evaluation' and sha(path)==receipt['metrics_sha256'],'Evaluation incomplete/changed')
            require(values['manifest_sha256']==protocol['manifest_sha256'] and values['checkpoint_metadata']['intervention_step']==6000,
                    'Wrong visualization endpoint or manifest')
            require(values['checkpoint_sha256']==sha(item['checkpoint']),f'{name}: display checkpoint mismatch')
            evaluations[name,endpoint]=path.parent
    sources,stills=[],[]
    def images_for(camera,frame,teacher=False):
        obs=observation(manifest,camera,frame)
        hr_path=args.manifest.parent/obs['hr_path']
        hr,digest=checked_image(hr_path,obs['hr_sha256'])
        images={'True HR':hr}
        sources.append({'role':'HR','camera':camera,'frame':frame,'path':str(hr_path),'sha256':digest})
        endpoint='train_fixed_6000' if obs['split']=='train' else 'test_6000'
        for name,item in methods.items():
            path=evaluations[name,endpoint]/'predictions'/camera/f'{frame:04d}.png'
            image,digest=checked_image(path)
            require(image.size==hr.size,f'Prediction size mismatch: {path}')
            images[item['label']]=image
            sources.append({'role':name,'camera':camera,'frame':frame,'path':str(path),'sha256':digest})
        if teacher:
            require(obs['split']=='train','Never load a held-out teacher')
            matches=[r for r in teacher_config['teacher_inputs'] if (r['camera'],r['frame'])==(camera,frame)]
            require(len(matches)==1,'Recorded train teacher missing')
            image,digest=checked_image(matches[0]['path'],matches[0]['sha256'])
            require(image.size==hr.size,'Teacher size mismatch')
            images['Frozen SwinIR (train)']=image
            sources.append({'role':'train_teacher','camera':camera,'frame':frame,'path':matches[0]['path'],'sha256':digest})
        return images,obs
    for camera,boxes in protocol['regions_by_camera_xyxy_exclusive'].items():
        for frame in protocol['fixed_still_frames']:
            images,obs=images_for(camera,frame,teacher=camera=='cam02')
            folder=args.out/f'{obs["split"]}_{camera}_frame{frame:04d}'
            folder.mkdir(exist_ok=False)
            comparison_panel(images,f'{manifest["scene"]} {obs["split"]} {camera} frame{frame}; display resized').save(folder/'overview.png')
            for index,(label,image) in enumerate(images.items()):
                image.save(folder/f'full_{index:02d}.png')
            files={}
            for name,box in boxes.items():
                panel=Image.new('RGB',(168*len(images),168+50),'white');draw=ImageDraw.Draw(panel)
                draw.text((4,4),f'{camera} frame{frame} {name}; native168px; fixed coordinates',fill='black')
                for index,(label,image) in enumerate(images.items()):
                    draw.text((index*168+4,29),label,fill='black')
                    panel.paste(image.crop(box),(index*168,50))
                panel.save(folder/f'{name}.png');files[name]=str(folder/f'{name}.png')
            stills.append({'camera':camera,'frame':frame,'split':obs['split'],'overview':str(folder/'overview.png'),
                           'native_rois':files,'full_image_labels_by_index':list(images)})
    frames=args.out/'video_frames';frames.mkdir(exist_ok=False)
    for frame in protocol['video_frames']:
        images,_=images_for(protocol['video_camera'],frame,teacher=False)
        panel=comparison_panel(images,f'{manifest["scene"]} {protocol["video_camera"]} frame{frame}; same time; display resized')
        # yuv420p requires even dimensions; padding does not crop scene content.
        if panel.width%2 or panel.height%2:
            even=Image.new('RGB',(panel.width+panel.width%2,panel.height+panel.height%2),'white');even.paste(panel,(0,0));panel=even
        panel.save(frames/f'{frame:04d}.png')
    legacy=module_from('soft_views_legacy_video',OLD/'evaluate.py')
    video=args.out/'comparison_cam00_60frames.mp4'
    video_receipt=legacy.encode_video(frames,video,protocol['playback_fps'])
    require(video_receipt['returncode']==0 and video.is_file(),f'Video encoding failed: {video_receipt}')
    result={'status':'completed_fixed_views','scene':manifest['scene'],'roi_protocol':str(args.roi_protocol),
            'roi_protocol_sha256':sha(args.roi_protocol),'methods_mapping':str(args.methods),'methods_mapping_sha256':sha(args.methods),
            'stills':stills,'video':video_receipt,'video_sha256':sha(video),'video_frames':60,
            'source_images':sources,'labels':{n:i['label'] for n,i in methods.items()},'source_sha256':sha(__file__),
            'precision':'Native PNG stills/crops and display-resized video; these quantized previews do not replace float-render metrics.',
            'information_boundary':'No model/teacher generation. HR for evaluation/display; cached teacher shown only at traincam02. Fixed ROIs precede prediction reads.',
            'playback_note':'15 playback frames/second is a registered visualization rate, not a timing-based end-to-end performance claim.',
            'finished_utc':datetime.now(timezone.utc).isoformat(),'parameter_updates':0,'gpu_used':False}
    write(args.out/'views.json',result)
    lines=[f'# {manifest["scene"]}：固定图像与60帧对照', '',
           'ROI 仅依据参考 HR 提前登记，不随方法结果更改；坐标未跟踪运动，名称仅描述参考帧内容。不同相机单独登记，不宣称精确物理对应。训练图额外显示已缓存教师，测试图不使用测试教师。', '',
           f'[60帧同相机同时间对照视频]({video})', '']
    for item in stills:
        lines += [f'{item["split"]} {item["camera"]}，帧{item["frame"]}：', '', f'![完整场景]({item["overview"]})','']
        for name,path in item['native_rois'].items():
            lines += [f'![{name}原生局部]({path})','']
    lines += ['视频为缩小展示，原生全图与168×168局部另存。量化展示不替代浮点指标；不存在由这些图自动确认的方法收益。','']
    (args.out/'README.md').write_text('\n'.join(lines))
    write(args.out/'complete.json',{'status':'completed_fixed_views','views_sha256':sha(args.out/'views.json'),'parameter_updates':0,'gpu_used':False})
    print(json.dumps({'out':str(args.out),'stills':len(stills),'video':str(video),'gpu_used':False}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--methods',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prepare-rois-only',action='store_true')
    parser.add_argument('--roi-protocol',type=Path)
    parser.add_argument('--roi-spec',type=Path,help='Optional HR-chosen camera->named168px boxes JSON, preparation only')
    args=parser.parse_args();args.manifest=args.manifest.resolve();args.out=args.out.resolve()
    if args.methods:args.methods=args.methods.resolve()
    if args.roi_protocol:args.roi_protocol=args.roi_protocol.resolve()
    try:
        prepare(args) if args.prepare_rois_only else export(args)
    except BaseException:
        args.out.mkdir(parents=True,exist_ok=True)
        write(args.out/'failed.json',{'status':'failed','traceback':traceback.format_exc()})
        raise


if __name__=='__main__':main()
