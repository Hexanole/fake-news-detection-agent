from pathlib import Path
import random
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

real_templates = [
    "Le ministère annonce {event} selon un communiqué officiel.",
    "Plusieurs médias reconnus rapportent {event} avec des sources vérifiables.",
    "Une conférence de presse officielle confirme {event}.",
    "L'agence de presse publie un article détaillé concernant {event}.",
    "Les autorités locales confirment {event} dans un rapport public.",
    "Un organisme officiel publie des données sur {event}.",
    "Une institution reconnue présente les résultats concernant {event}.",
    "Le journal publie une analyse factuelle sur {event}.",
    "Le communiqué officiel indique {event} sans éléments sensationnels.",
    "Des experts cités par des médias fiables expliquent {event}.",
]

fake_templates = [
    "Urgent choc {event} les médias cachent la vérité partagez avant suppression.",
    "Scandale incroyable {event} une source anonyme révèle un secret.",
    "Breaking alerte {event} aucune source officielle ne veut en parler.",
    "Vérité cachée {event} cette information va disparaître bientôt.",
    "Partagez vite {event} avant que les autorités suppriment tout.",
    "On dit que {event} mais personne ne montre de preuve officielle.",
    "Une rumeur affirme {event} sans document ni source claire.",
    "Exclusif secret {event} les médias refusent de le montrer.",
    "Catastrophe énorme {event} selon une source anonyme non confirmée.",
    "Alerte virale {event} tout le monde doit partager maintenant.",
]

events_real = [
    "une nouvelle mesure économique",
    "une réunion gouvernementale",
    "un changement dans les transports publics",
    "une publication de statistiques nationales",
    "une conférence scientifique",
    "une décision administrative",
    "une opération de maintenance urbaine",
    "une campagne de sensibilisation",
    "une annonce météorologique",
    "un rapport annuel de la banque centrale",
    "un accord signé entre institutions",
    "une modification du calendrier scolaire",
    "une mise à jour sanitaire officielle",
    "une déclaration du porte-parole",
    "une publication universitaire",
]

events_fake = [
    "une explosion énorme",
    "un complot mondial",
    "une attaque secrète",
    "une catastrophe cachée",
    "une interdiction totale imminente",
    "un danger que personne ne veut révéler",
    "un scandale jamais vu",
    "une vérité que les médias censurent",
    "un événement mystérieux",
    "une manipulation massive",
    "une découverte interdite",
    "une alerte extrême",
    "une crise cachée",
    "un plan secret",
    "une preuve supprimée",
]

rows = []

for _ in range(250):
    template = random.choice(real_templates)
    event = random.choice(events_real)
    rows.append({
        "text": template.format(event=event),
        "label": "real"
    })

for _ in range(250):
    template = random.choice(fake_templates)
    event = random.choice(events_fake)
    rows.append({
        "text": template.format(event=event),
        "label": "fake"
    })

random.shuffle(rows)

df = pd.DataFrame(rows)
output_path = DATA_DIR / "train_text.csv"
df.to_csv(output_path, index=False, encoding="utf-8")

print(f"Dataset créé : {output_path}")
print(df["label"].value_counts())