# Deploying Grisz (not done yet)

Nothing here has been run. Each step that can cost money says so. Do not start before every box in "Before deploying" is ticked.

## Before deploying
- [ ] Full test run on the PC: page, access codes, uploads, quizzes, Grafana, Ragas, Langfuse
- [ ] Redesign + debug pass (owner's rule)
- [ ] Privacy notice and terms on the site: questions and module text go to Groq (and Langfuse if tracing is on); **students' names and scores are stored** (`storage/results.db`); who runs the site; how to ask for deletion (teacher uses `DELETE /results/{name}`)
- [ ] Long, different `TEACHER_CODE` and `STUDENT_CODE`
- [ ] Only openly licensed or your own PDFs in `data/`

## Server: Oracle Cloud Always Free (recommended fit, ~810 MB needed)
1. Sign up at oracle.com/cloud/free. **Needs a card for identity checks.** Stay on Always Free resources only.
2. **First:** Billing & Cost Management → Budgets → create a budget of USD 1 with an email alert at 1%. Any charge at all then emails you.
3. Create a VM: Ampere A1 (Arm), 2 OCPU, 12 GB RAM, Ubuntu. Marked "Always Free eligible". If it says "out of capacity", try later or another availability domain.
4. In the VM's network security list, allow inbound TCP 80 and 443. Nothing else (SSH 22 is already open).
5. Oracle's Ubuntu images also block ports inside the VM. On the VM:
   ```bash
   sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
   sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
   sudo netfilter-persistent save
   ```

## Domain (free)
6. duckdns.org → sign in → make a subdomain (e.g. `grisz`) → set it to the VM's public IP.

## Run it
7. On the VM: install Docker (docs.docker.com/engine/install/ubuntu), then:
   ```bash
   git clone https://github.com/dioys/grisz.git && cd grisz
   nano .env        # same as on the PC, plus DOMAIN=grisz.duckdns.org
   mkdir data       # copy the PDFs here (scp), or upload them later on the page
   docker compose -f compose.yaml -f compose.prod.yaml up -d --build
   ```
8. Open https://grisz.duckdns.org. Caddy gets the HTTPS certificate by itself on the first visit.

## Grafana and Prometheus on the server
They are not on the internet. From the PC: `ssh -L 3000:127.0.0.1:3000 ubuntu@<VM IP>`, then open http://127.0.0.1:3000.

## What protects it
- Only Caddy (80/443) is public; `/metrics` returns 404 from outside.
- Access codes on every endpoint; 30 requests per minute per visitor (`RATE_LIMIT_PER_MINUTE`).
- Uploads capped at 50 MB (app) and 55 MB (Caddy).
