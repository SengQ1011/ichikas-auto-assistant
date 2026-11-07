from kotonebot import task

from iaa.tasks.common import go_home
from iaa.context import task_reporter
from .live import ListLoopPlan, SingleLoopPlan, solo_live as do_solo_live
from iaa.i18n import TStr


@task('自动演出')
def auto_live(plan: SingleLoopPlan | ListLoopPlan) -> None:
    reporter = task_reporter()
    reporter.message(TStr(zh_CN='准备自动演出参数', en_US='Preparing auto live options'))
    
    if plan.loop_count is not None and plan.loop_count <= 0:
        raise ValueError('loop_count 必须为正整数或 None。')
    if isinstance(plan, SingleLoopPlan) and plan.song_select_mode == 'specified' and not plan.song_name:
        raise ValueError('song_name is required when song_select_mode is specified.')
    if isinstance(plan, ListLoopPlan) and plan.play_mode == 'hold_center':
        raise ValueError('hold_center only supports single-song loop.')

    reporter.message(TStr(zh_CN='返回首页准备进入演出', en_US='Returning home before live'))
    # 游戏没开时自动启动（单独运行自动演出时不会经过「启动游戏」任务）
    if isinstance(plan, SingleLoopPlan) and plan.play_mode == 'hold_center':
        from .hold_center import go_home_before_start
        go_home_before_start()  # 同上，另外超过 3 分钟还没到首页就停止
    else:
        go_home(check_alive=True)
    reporter.message(TStr(zh_CN='进入自动演出流程', en_US='Starting auto live flow'))
    do_solo_live(plan)
