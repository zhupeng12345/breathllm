"""BreathLLM 的数据准备、轨道 A 复现及研究入口。"""

import argparse
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "v3wave/代码"), str(ROOT / "breath5")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = parser.add_subparsers(dest='task',required=True)
    tasks.add_parser('prepare',help='校验并重建开发受试者因果标签')
    tasks.add_parser('weights',help='校验锁定版本的模型文件')
    smoke = tasks.add_parser('smoke',help='运行合成数据测试或有限真实前向')
    smoke.add_argument('--real',action='store_true')
    smoke.add_argument('--device',default='cuda')
    t03 = tasks.add_parser('t03',help='重建 T03 训练和整夜评分归档')
    t03.add_argument('--folds',nargs='+',type=int,choices=(1,2,3),default=(1,))
    t03.add_argument('--seeds',nargs='+',type=int,choices=(4121,4122,4123),default=(4121,))
    t03.add_argument('--device',default='cuda')
    for task,help_text in (('r0','T03 瓶颈和精度审计'),('r1','运行原定暂停排序与校准探针'),
                           ('r2','基于 T03 归档的三分支归因'),('pael','复现原定持续性异常事件探针')):
        entry = tasks.add_parser(task,help=help_text)
        entry.add_argument('--device',default='cuda')
        if task=='r0':
            entry.add_argument('--statistics-only',action='store_true')
        if task=='pael':
            entry.add_argument('--steps',type=int,choices=(0,128,512),default=128)
    if sys.argv[1:2] == ['baseline']:
        return importlib.import_module('60s五分类baseline').main(sys.argv[2:])
    tasks.add_parser('baseline',help='20 种本地基线，使用 baseline --help 查看运行参数')
    args = parser.parse_args()
    import 发布复现 as repro
    repro.DEVICE = getattr(args,'device','cuda')
    try:
        if args.task=='prepare':
            repro.prepare()
        elif args.task=='weights':
            print(f"已核验 {len(repro.weights())} 个权重和模型配置文件")
        elif args.task=='smoke':
            if args.real:
                repro.real_smoke(args.device)
            else:
                sys.path.insert(0,str(ROOT/'tests'))
                import smoke
                smoke.run()
        elif args.task=='t03':
            repro.train_t03(args.folds,args.seeds,args.device)
        else:
            names = {'r0':'BreathLLM-R0','r1':'BreathLLM-R1','r2':'BreathLLM-R2','pael':'BreathLLM-R3-PAEL'}
            entry = importlib.import_module(names[args.task])
            sys.argv[1:] = []
            if args.task=='pael':
                sys.argv[1:] = ['--steps',str(args.steps)]
            if args.task=='r0' and args.statistics_only:
                sys.argv[1:] = ['--statistics-only']
            entry.main() if args.task!='pael' else entry.run_main()
    except (FileNotFoundError,ValueError) as error:
        parser.exit(2,f'{error}\n')


if __name__=='__main__':
    main()
