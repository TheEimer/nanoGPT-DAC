import time
import torch 
from model import GPTConfig, GPT
from contextlib import nullcontext
from train import get_batch, estimate_loss

def load_checkpoint(checkpoint_name, model_args, optimizer_args, device):
    # load the checkpoint
    print(f"Resuming training from {checkpoint_name}")
    # resume training from a checkpoint.
    checkpoint = torch.load(checkpoint_name, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    optimizer = model.configure_optimizers(optimizer_args["weight_decay"], optimizer_args["learning_rate"], (optimizer_args["beta1"], optimizer_args["beta2"]), optimizer_args["device_type"])
    optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None
    return model, optimizer, iter_num, best_val_loss

# TODO: make hydra-able
def do_step(agent, agent_steps, checkpoint_name, device, train_args, model_args, optimizer_args, scaler, save_file, ddp, ptdtype):
    model, optimizer, iter_num, best_val_loss = load_checkpoint(checkpoint_name, model_args, optimizer_args, device)
    raw_model = model.module if ddp else model
    ctx = nullcontext() if 'cpu' in device else torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    # TODO: define state representation
    state = None

    for _ in range(agent_steps):
        # get next lr
        lr = agent.predict(state)

        # apply learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(train_args["gradient_accumulation_steps"]):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == train_args["gradient_accumulation_steps"] - 1)
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / train_args["gradient_accumulation_steps"] # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if train_args["grad_clip"] != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_args["grad_clip"])
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
        reward = -loss.item()

        if loss.item() > 1.1 * best_val_loss:
            print(f"loss {loss.item()} is 10% higher than best val loss {best_val_loss}, falling back to previous best checkpoint.")
            model, optimizer, _, _ = load_checkpoint(checkpoint_name)
            reward -= 100

        # TODO: implement agent update
        # On second thought: this should only collect and not update?
        # Or do we maybe do reptile here?
        agent.update(state=state, action=lr, reward=reward)
        # TODO: implement state transition
        next_state = None
        state = next_state

    # evaluate the loss on train/val sets and write checkpoints
    losses = estimate_loss()
    print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    # save checkpoint
    best_val_loss = losses['val']
    if iter_num > 0:
        checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
        print(f"saving checkpoint to {save_file}")
        torch.save(checkpoint, save_file)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    lossf = loss.item() * train_args["gradient_accumulation_steps"]
    mfu = raw_model.estimate_mfu(train_args["batch_size"] * train_args["gradient_accumulation_steps"], dt)
    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
    print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    return iter_num, losses['val']