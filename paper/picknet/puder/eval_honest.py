"""Re-score the released puder v1 model on its own 1024 test drafts with honest accounting + header ablations."""
import torch, configuration, runner
torch.manual_seed(0)
cfg = configuration.Config()
cfg.testdata_only = True; cfg.checkpoint_load = True; cfg.training = False
cfg.checkpoint_load_path = 'models/puder_v1_7M.pt'; cfg.device = 'cpu'; cfg.silent = True; cfg.tensorboard = False; cfg._args_config = {}
r = runner.Runner(cfg); r.init(); r.load_dataset(); r.load_model()
picks, packs, rates = r.dataset_test.tensors
H = 8
tl = r.tokens.token_list
evt = [tl[i] for i in picks[:, 1].tolist()]

def score(p, skip_packs=False):
    with torch.no_grad():
        lg, _ = r.compute_logits(p, packs, skip_packs=skip_packs)
    pred = lg[:, H-1:, :].argmax(-1)
    tgt = p[:, H:].long()
    hit = pred == tgt
    real = tgt != 0
    forced = (packs != 0).sum(-1) == 1
    return hit, real, forced

def report(name, p, skip=False):
    hit, real, forced = score(p, skip)
    reported = hit.float().mean().item()          # puder's formula: every slot, padding included
    honest = hit[real].float().mean().item()
    nonforced = hit[real & ~forced].float().mean().item()
    prem = torch.tensor([e == '$PremierDraft' for e in evt])
    pr = hit[prem][real[prem]].float().mean().item()
    tr = hit[~prem][real[~prem]].float().mean().item()
    print(f'| {name} | {reported:.4f} | {honest:.4f} | {nonforced:.4f} | {pr:.4f} | {tr:.4f} |')

print(f'drafts: {len(picks)}, premier {sum(e=="$PremierDraft" for e in evt)}, trad {sum(e=="$TradDraft" for e in evt)}')
print('| variant | reported formula (pads count) | real picks | non-forced | Premier | Trad |\n|---|---|---|---|---|---|')
report('as released', picks)
def shuf(cols, seed=0):
    p = picks.clone(); g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(p), generator=g)
    for c in cols: p[:, c] = picks[perm, c]
    return p
report('shuffle event W/L across drafts', shuf([6, 7]))
report('shuffle event W/L (seed 1)', shuf([6, 7], 1))
p = picks.clone(); p[:, 6] = r.tokens.index['$+3']; p[:, 7] = r.tokens.index['$-3']
report('W/L fixed to 3-3', p)
report('shuffle skill (games, win-rate buckets)', shuf([4, 5]))
report('shuffle rank', shuf([3]))
report('shuffle month', shuf([2]))
report('shuffle W/L + skill + rank', shuf([3, 4, 5, 6, 7]))
report('no pack contents (skip_packs)', picks, True)
# wins vs accuracy
hit, real, _ = score(picks)
w = torch.tensor([int(tl[i][2:]) for i in picks[:, 6].tolist()])
for k in range(8):
    s = w == k
    if s.sum(): print(f'wins={k}: n={s.sum().item()} acc={hit[s][real[s]].float().mean():.3f}')
print('per expansion (Premier only):')
ex = [tl[i] for i in picks[:, 0].tolist()]
prem = [e == '$PremierDraft' for e in evt]
for e in sorted(set(ex)):
    s = torch.tensor([a == e and b for a, b in zip(ex, prem)])
    if s.sum(): print(f'  {e}: n={s.sum().item()} acc={hit[s][real[s]].float().mean():.3f}')
