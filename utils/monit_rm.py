import visdom
import numpy as np

windows = {}
windows['ip'] ='207.148.26.49'
windows['port'] = 8097
windows['port'] = 8090
windows['session'] = "DefaultSession"

summary_nll_gen = []
summary_bleu_2 = []
summary_bleu_3 = []
summary_bleu_4 = []
summary_bleu_5 = []
summary_temp = []
summary_text = {}
f_loss = []
f_acc = []

f_adv_loss = []
f_adv_acc = []
f_adv_loss_eval = []
f_adv_acc_eval = []

loss_lines = {}
acc_lines = {}

def visualize_scores(scores):
    global ummary_nll_gen, summary_bleu_2, summary_bleu_3, summary_bleu_4, summary_bleu_5
    try:
        summary_nll_gen.append(scores[1])
        vision_update(summary_nll_gen, "NLL_GEN")

        summary_bleu_2.append(scores[0][0])
        vision_update(summary_bleu_2, "BLEU-2")

        summary_bleu_3.append(scores[0][1])
        vision_update(summary_bleu_3, "BLEU-3")

        summary_bleu_4.append(scores[0][2])
        vision_update(summary_bleu_4, "BLEU-4")

        summary_bleu_5.append(scores[0][3])
        vision_update(summary_bleu_5, "BLEU-5")
    except:
        pass
        
# def visualize_line(loss, acc):
#     global f_loss, f_acc
#     f_loss.append(loss)
#     f_acc.append(acc)
#     vision_update(f_loss, "Training F Loss")
#     vision_update(f_acc, "Training F Accuracy")

def visualize_append_single_line(data, name):
    global acc_lines
    if name not in acc_lines:
        acc_lines[name] = []
    acc_lines[name].append(data)
    vision_update(acc_lines[name], "ACCURACY-{}".format(name))

def visualize_append_line(loss, acc, name):
    global loss_lines, acc_lines
    if name not in loss_lines:
        loss_lines[name] =  []
        acc_lines[name] = []

    loss_lines[name].append(loss)
    acc_lines[name].append(acc)
    vision_update(loss_lines[name], "LOSS-{}".format(name))
    vision_update(acc_lines[name], "ACCURACY-{}".format(name))

    # global f_adv_loss, f_adv_acc
    # if name == "TRAIN-SET":
    #     f_adv_loss.append(loss)
    #     f_adv_acc.append(acc)
    #     vision_update(f_adv_loss, "Adversarial F Loss-{}".format(name))
    #     vision_update(f_adv_acc, "Adversarial F Accuracy-{}".format(name))
    # elif name == "DEV-SET":
    #     f_adv_loss_eval.append(loss)
    #     f_adv_acc_eval.append(acc)
    #     vision_update(f_adv_loss_eval, "Adversarial F Loss-{}".format(name))
    #     vision_update(f_adv_acc_eval, "Adversarial F Accuracy-{}".format(name))

def visualize_generated_text(text, name):
    global summary_text
    if name not in summary_text:
        summary_text[name] = ""
    summary_text[name] = "{}\n\n{}".format(summary_text[name], text).replace('\n','</br>').replace('<unk>', '**unk**').replace('<pad>', '_')
    vision_append_text(summary_text[name], "Generated Text - {}".format(name))

def visualize_temperature(temp, update_dashboard=False):
    global summary_temp
    summary_temp.append(temp)
    if update_dashboard:
        vision_update(summary_temp, "Temperature")


def generate_session_name(args):
    import time
    t0 = time.time()
    session = "{}-{}Content{}_PreTr{}_{}_Adapt{}_F-{}-{}-{}-{}-{}_H-{}-{}-{}_maxsn{}_vs{}_bs{}_sl{}_advLog{}_temp{}_Gstep{}_Dstep{}_G-{}-{}-{}-{}-{}_D-{}-{}-{}_N{}_T{}".format(
                args.user, args.dataset, args.train_with_content, args.load_gen_pretrain, args.attack_mode, args.if_adapt_alpha,
                args.max_num_comment, args.max_num_comment_test, args.g_f_alpha, args.g_f_step, args.f_batch_size,
                args.g_h_alpha, args.g_h_step, args.h_batch_size,
                args.max_num_data, args.vocab_size, args.batch_size, args.max_seq_len,
				args.adv_log_step, args.temperature, args.ADV_g_step, args.ADV_d_step,
				args.gen_embed_dim, args.gen_hidden_dim,
				args.mem_slots, args.num_heads, 
				args.head_size, args.dis_embed_dim, 
				args.dis_hidden_dim, args.num_rep, args.ADV_train_epoch, t0)
    return session


def getvis():
    return windows['vis']

def vision_save_env():
    try:
        windows['vis'].save([windows['session']])
    except:
        pass

def set_session(session, ip):
    windows['session'] = session
    windows['ip'] = ip
    windows['vis'] = visdom.Visdom(server=windows['ip'], port=windows['port'], env=windows['session'])

def vision_change_ip(ip):
    windows['ip'] = ip
    windows['vis'] = visdom.Visdom(server=windows['ip'], port=windows['port'], env=windows['session'])

def vision_change_port(port):
    windows['port'] = port
    windows['vis'] = visdom.Visdom(server=windows['ip'], port=windows['port'], env=windows['session'])

def vision_line(data, name):
    try:
        a = np.asarray(data)
        vis = windows['vis']
        w = vis.line(X=np.array(range(len(data))), Y=a, opts=dict(title="{}".format(name, windows['session']), webgl=False))
        windows[name] = w
    except:
        pass

def vision_append_text(str, name):
    try:
        vis = windows['vis']
        if name not in windows:
            w = vis.text(str, opts=dict(title="{}".format(name, windows['session'])))
        else:
            w = vis.text(str, win=windows[name], opts=dict(title="{}".format(name, windows['session'])))
        windows[name] = w
    except:
        pass

def vision_update(data, name):
    a = np.asarray(data)
    try:
        vis = windows['vis']
        if name not in windows:
            vision_line([0], name)
        vis.line(X=np.array(range(len(data))), Y=a, win=windows[name],
                 opts=dict(title="{}".format(name, windows['session']), webgl=False))
    except:
        pass
