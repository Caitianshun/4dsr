"""Frozen native ROIs and both complete held-out camera videos, CPU-only."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from PIL import Image,ImageDraw
from summarize import read,write,sha

ROOT=Path(__file__).resolve().parents[2]


def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser()
    for k in ['manifest','methods','roi-protocol','teacher-index','out']:p.add_argument('--'+k,required=True,type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    m=read(a.manifest);methods=read(a.methods)['methods'];roi=read(a.roi_protocol);assert roi['manifest_sha256']==sha(a.manifest)
    teacher={(r['camera'],r['frame']):r for r in read(a.teacher_index)['entries']}
    old=module('original_fixed_view_helpers',ROOT/'experiments/dynamic_sr_detail_supervision_20260924/export_views.py')
    legacy=module('original_video_helpers',ROOT/'experiments/dynamic_sr_20260918/evaluate.py')
    sources=[];validated={}
    for name,item in methods.items():
        checkpoint_sha=sha(item['checkpoint'])
        for split,folder in item['evaluations'].items():
            folder=Path(folder);receipt=read(folder/'complete.json');metrics=read(folder/'metrics.json')
            assert receipt['status']=='completed_evaluation' and receipt['metrics_sha256']==sha(folder/'metrics.json')
            assert metrics['checkpoint_sha256']==checkpoint_sha
            validated[name,split]=folder
    def images(names,camera,frame):
        record=next(r for r in m['observations'] if r['camera_id']==camera and r['frame_index']==frame)
        result={}
        for label,key in [('True HR','hr'),('Observed LR (display bilinear)','lr')]:
            path=a.manifest.parent/record[key+'_path'];assert sha(path)==record[key+'_sha256']
            im=Image.open(path).convert('RGB');result[label]=im if key=='hr' else im.resize(tuple(m['resolutions']['hr']),Image.Resampling.BILINEAR)
        split='train_fixed' if record['split']=='train' else record['split']
        for name in names:
            folder=validated[name,split]
            path=folder/'predictions'/camera/f'{frame:04d}.png'
            result[name]=Image.open(path).convert('RGB');sources.append(dict(method=name,camera=camera,frame=frame,path=str(path),sha256=sha(path)))
        if record['split']=='train':
            entry=teacher[camera,frame];path=a.manifest.parent/entry['relative_path'];assert sha(path)==entry['sha256']
            result['Frozen SwinIR (train)']=Image.open(path).convert('RGB')
        return result
    groups={'teacher':['M0','M1'],'long_baseline':['U6000','U18000','U40000','W6000','W18000','W40000'],
        'systems':['M0','M1','U40000','W40000','B4','G']}
    artifacts=[]
    for group,names in groups.items():
        folder=a.out/group;folder.mkdir()
        for camera,boxes in roi['regions_by_camera_xyxy_exclusive'].items():
            for frame in roi['fixed_still_frames']:
                imgs=images(names,camera,frame);dest=folder/f'{camera}_{frame:04d}';dest.mkdir()
                old.comparison_panel(imgs,f'{group} {camera} frame{frame}; display resized').save(dest/'overview.png')
                for i,(name,im) in enumerate(imgs.items()):im.save(dest/f'full_{i:02d}.png')
                for label,box in boxes.items():
                    panel=Image.new('RGB',(168*len(imgs),218),'white');draw=ImageDraw.Draw(panel)
                    draw.text((4,4),f'{camera} frame{frame} {label}; fixed native168px',fill='black')
                    for i,(name,im) in enumerate(imgs.items()):draw.text((168*i+3,28),name[:26],fill='black');panel.paste(im.crop(box),(168*i,50))
                    path=dest/(label+'.png');panel.save(path);artifacts.append(str(path))
        for camera in ['cam00','cam01']:
            frames=folder/(camera+'_video_frames');frames.mkdir()
            for frame in roi['video_frames']:
                panel=old.comparison_panel(images(names,camera,frame),f'{group} {camera} frame{frame}; matched time')
                if panel.width%2 or panel.height%2:
                    padded=Image.new('RGB',(panel.width+panel.width%2,panel.height+panel.height%2),'white');padded.paste(panel,(0,0));panel=padded
                panel.save(frames/f'{frame:04d}.png')
            video=folder/(camera+'_60frames.mp4');result=legacy.encode_video(frames,video,roi['playback_fps']);assert result['returncode']==0
            artifacts.append(str(video))
    write(a.out/'complete.json',dict(status='completed_fixed_views',artifacts=artifacts,sources=sources,parameter_updates=0,roi_sha256=sha(a.roi_protocol)))
    lines=['# Multi4D / Wu 固定图像与视频','', '所有ROI沿用原坐标，视频为15播放帧/秒，包含完整cam00与cam01各60帧。LR仅为显示上采样；评价使用浮点模型渲染。','']
    for path in artifacts:lines += [f'[{Path(path).parent.name}/{Path(path).name}]({path})','']
    (a.out/'README.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
