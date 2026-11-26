import math


def integer(x, lower, name):
    if type(x) is not int or x < lower: raise ValueError(name)
    return x


def rank_batches(order, batch_size=64, rank=0, world_size=2, start_batch=0, pad_empty=False):
    integer(batch_size,1,'batch_size'); integer(world_size,1,'world_size')
    integer(rank,0,'rank'); integer(start_batch,0,'start_batch')
    if rank >= world_size or type(pad_empty) is not bool: raise ValueError('rank/pad_empty')
    seq=list(order)
    for x in seq: integer(x,0,'sample_index')
    if len(seq) != len(set(seq)): raise ValueError('duplicate_order')
    total=(len(seq)+batch_size-1)//batch_size
    if start_batch>total: raise ValueError('cursor_out_of_bounds')
    result=[]
    for i in range(start_batch,total):
        chunk=seq[i*batch_size:(i+1)*batch_size]
        local=chunk[rank::world_size]; valid=len(local)
        if not local and pad_empty: local=[chunk[0]]
        result.append({'batch_index':i,'indices':local,'global_count':len(chunk),'valid_count':valid})
    return result


def loss_mean_multiplier(local_valid_count, global_count, world_size=2):
    integer(local_valid_count,0,'local_count'); integer(global_count,1,'global_count'); integer(world_size,1,'world_size')
    if local_valid_count>global_count:raise ValueError('local_count_exceeds_global')
    return world_size*local_valid_count/global_count


def lr_factor(step,steps_per_epoch=349,epochs=20,warmup_epochs=1,min_lr_ratio=0.01):
    integer(step,0,'step');integer(steps_per_epoch,1,'steps_per_epoch')
    integer(epochs,1,'epochs');integer(warmup_epochs,0,'warmup_epochs')
    if warmup_epochs>epochs or type(min_lr_ratio) not in (int,float) or not math.isfinite(min_lr_ratio) or not 0<=min_lr_ratio<=1:raise ValueError('lr_limits')
    warmup=steps_per_epoch*warmup_epochs;total=steps_per_epoch*epochs
    if step<warmup:return (step+1)/max(1,warmup)
    position=min(1,(step-warmup)/max(1,total-warmup))
    return min_lr_ratio+(1-min_lr_ratio)*(1+math.cos(math.pi*position))/2
