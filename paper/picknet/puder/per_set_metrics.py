"""Per-expansion top-1 / non-forced top-1 / NLL for the released puder v1 model on its own Premier test drafts."""
import json, torch, configuration, runner
torch.manual_seed(0)
cfg = configuration.Config()
cfg.testdata_only = True; cfg.checkpoint_load = True; cfg.training = False
cfg.checkpoint_load_path = 'models/puder_v1_7M.pt'; cfg.device = 'cpu'; cfg.silent = True; cfg.tensorboard = False; cfg._args_config = {}
r = runner.Runner(cfg); r.init(); r.load_dataset(); r.load_model()
picks, packs, rates = r.dataset_test.tensors
H = 8
tl = r.tokens.token_list
with torch.no_grad():
    lg, _ = r.compute_logits(picks, packs)
lg = lg[:, H-1:, :].float()
tgt = picks[:, H:].long()
lp = lg.log_softmax(-1)
nll = -lp.gather(-1, tgt[..., None]).squeeze(-1)
hit = lg.argmax(-1) == tgt
real = tgt != 0
forced = (packs != 0).sum(-1) == 1
prem = torch.tensor([tl[i] == '$PremierDraft' for i in picks[:, 1].tolist()])
ex = [tl[i] for i in picks[:, 0].tolist()]
out = {}
for e in sorted(set(ex)):
    s = torch.tensor([a == e for a in ex]) & prem
    if not s.sum(): continue
    rr, nf = real[s], real[s] & ~forced[s]
    out[e] = dict(drafts=int(s.sum()), top1=float(hit[s][rr].float().mean()), nf=float(hit[s][nf].float().mean()),
                  nll=float(nll[s][rr].mean()), forced_share=float(forced[s][rr].float().mean()))
    print(e, out[e])
json.dump(out, open('puder_per_set.json', 'w'), indent=1)
