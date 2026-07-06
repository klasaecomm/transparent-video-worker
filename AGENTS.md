# Zasady — instrukcja dla agentów AI

Ta aplikacja działa na klastrze **klasaecomm-heweliusza-node** i jest wdrażana przez
**Argo CD** z repozytorium infrastruktury (GitOps, `selfHeal` włączony).

## Twarde reguły — NIGDY nie łam

1. **Nie wdrażaj ręcznie.** `kubectl apply / set image / edit / scale` na zasobach tej
   aplikacji zostanie automatycznie cofnięte przez Argo CD w kilka minut.
2. **Deploy tylko przez pipeline:** build w Jenkinsie → obraz do registry z immutable
   tagiem → zmiana tagu w repo infrastruktury. Nigdy `:latest`.
3. **Sekrety w Vault** (`kv/kubernetes/<namespace>/...`) — nigdy w kodzie ani manifestach.
4. Manifesty Kubernetes tej aplikacji NIE są w tym repo — są w
   `klasaecomm-infrastructure/apps/transparent-video-worker/`.

## Pełne zasady i kontekst systemu (źródło prawdy)

Kompletne reguły, struktura klastra i runbooki żyją w repo infrastruktury.
Pobierz zawsze aktualną wersję:

```bash
gh api repos/klasaecomm/klasaecomm-infrastructure/contents/AGENTS.md \
  --jq '.content' | base64 -d
```

Zanim ruszysz z wdrożeniem, konfiguracją, sekretami lub infrastrukturą — przeczytaj ten plik.
