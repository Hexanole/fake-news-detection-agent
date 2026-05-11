import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update, BotCommand
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from agent.pipelines.verify_news_pipeline import VerifyNewsPipeline
from agent.storage import HistoryStorage
from src.config import REPORTS_DIR

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

pipeline = VerifyNewsPipeline()
history_storage = HistoryStorage()

# Dernier rapport par chat Telegram
LAST_REPORT_BY_CHAT: dict[int, str] = {}


def extract_source_from_text(text: str) -> tuple[str, str]:
    if not text:
        return "", ""

    lowered = text.lower()
    markers = ["source:", "source :", "url:", "url :", "lien:", "lien :"]

    for marker in markers:
        index = lowered.find(marker)
        if index != -1:
            main_text = text[:index].strip()
            source = text[index + len(marker):].strip()
            return main_text, source

    return text.strip(), ""


def format_report(report: dict, detailed: bool = False) -> str:
    decision = report.get("decision", {})
    signals = report.get("signals", {})
    reasons = decision.get("reasons", [])

    message = "🧠 Disinfo Agent Nissan\n\n"
    message += f"📊 Score risque : {decision.get('risk_score')}\n"
    message += f"🚦 Niveau : {decision.get('risk_level')}\n"
    message += f"🏷️ Label : {decision.get('label')}\n\n"

    message += "🔎 Raisons principales :\n"
    for reason in reasons[:6]:
        message += f"- {reason}\n"

    if detailed:
        text_analysis = signals.get("text_analysis", {})
        ml_text = signals.get("ml_text_model", {})
        source_rep = signals.get("source_reputation", {})
        web = signals.get("web_search", {})
        image = signals.get("image_analysis", {})

        message += "\n📌 Détails techniques :\n"
        message += f"- Texte : {text_analysis.get('label')} | score={text_analysis.get('score')}\n"
        message += f"- Modèle IA texte : {ml_text.get('label')} | fake_probability={ml_text.get('fake_probability')}\n"
        message += f"- Source : {source_rep.get('domain')} | {source_rep.get('label')}\n"
        message += f"- Articles web trouvés : {web.get('articles_found')}\n"
        message += f"- Image analysée : {image.get('success')}\n"

    message += "\n⚠️ Résultat = estimation du risque, pas vérité absolue."
    return message


async def post_init(application):
    await application.bot.set_my_commands([
        BotCommand("start", "Démarrer le bot"),
        BotCommand("help", "Afficher l'aide"),
        BotCommand("status", "Tester l'état de l'agent"),
        BotCommand("verify", "Analyser une news"),
        BotCommand("history", "Afficher les dernières analyses"),
        BotCommand("lastreport", "Télécharger le dernier rapport JSON"),
    ])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = (
        "Salam 👋\n\n"
        "Je suis Nissan, ton agent de vérification de fake news.\n\n"
        "✅ Tu peux envoyer directement une news.\n"
        "✅ Tu peux envoyer une image avec une légende.\n"
        "✅ Tu peux ajouter une source avec : Source: https://site.com\n\n"
        "Exemple :\n"
        "Urgent choc explosion énorme... Source: https://example.com\n\n"
        "Commandes :\n"
        "/verify votre news Source: https://site.com\n"
        "/history\n"
        "/lastreport\n"
        "/status"
    )
    await update.message.reply_text(message)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = (
        "📌 Mode d'emploi\n\n"
        "1) Analyse directe :\n"
        "Envoie une news comme message normal.\n\n"
        "2) Analyse avec commande :\n"
        "/verify Urgent une explosion... Source: https://site.com\n\n"
        "3) Analyse image :\n"
        "Envoie une image avec une légende contenant la news.\n\n"
        "4) Historique :\n"
        "/history\n\n"
        "5) Dernier rapport JSON :\n"
        "/lastreport"
    )
    await update.message.reply_text(message)


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = (
        "✅ Agent actif\n"
        "✅ Pipeline chargé\n"
        "✅ Historique disponible\n"
        "✅ Bot Telegram connecté\n\n"
        "Tu peux envoyer une news maintenant."
    )
    await update.message.reply_text(message)


async def run_verification(
    update: Update,
    text: str,
    image_path: str = "",
    source: str = "",
    detailed: bool = False
):
    chat_id = update.effective_chat.id

    if not text and not image_path:
        await update.message.reply_text(
            "Donne-moi une news ou une image à analyser.\n\n"
            "Exemple :\n"
            "/verify Urgent explosion... Source: https://site.com"
        )
        return

    await update.message.reply_text("⏳ Analyse en cours...")

    try:
        report = pipeline.verify(
            text=text,
            image_path=image_path,
            source=source,
        )

        LAST_REPORT_BY_CHAT[chat_id] = report.get("report_path", "")

        await update.message.reply_text(format_report(report, detailed=detailed))

    except Exception as e:
        await update.message.reply_text(f"❌ Erreur pendant l'analyse : {e}")


async def verify_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    raw_text = " ".join(context.args)

    news_text, source = extract_source_from_text(raw_text)

    await run_verification(
        update=update,
        text=news_text,
        image_path="",
        source=source,
        detailed=True
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text or ""
    news_text, source = extract_source_from_text(user_text)

    await run_verification(
        update=update,
        text=news_text,
        image_path="",
        source=source,
        detailed=False
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    caption = update.message.caption or ""
    news_text, source = extract_source_from_text(caption)

    await update.message.reply_text("📷 Image reçue. Téléchargement...")

    try:
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)

        temp_dir = Path(tempfile.gettempdir()) / "disinfo_agent_telegram"
        temp_dir.mkdir(parents=True, exist_ok=True)

        image_path = temp_dir / f"{photo.file_unique_id}.jpg"
        await file.download_to_drive(custom_path=str(image_path))

        await run_verification(
            update=update,
            text=news_text,
            image_path=str(image_path),
            source=source,
            detailed=True
        )

    except Exception as e:
        await update.message.reply_text(f"❌ Erreur image : {e}")


async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = history_storage.read_last(limit=5)

    if not rows:
        await update.message.reply_text("Aucun historique pour le moment.")
        return

    message = "📜 Dernières analyses :\n\n"

    for i, row in enumerate(rows, start=1):
        text = row.get("text", "")
        short_text = text[:80] + "..." if len(text) > 80 else text

        message += f"{i}) {row.get('timestamp')}\n"
        message += f"   Niveau : {row.get('risk_level')} | Score : {row.get('risk_score')}\n"
        message += f"   Source : {row.get('domain')}\n"
        message += f"   Texte : {short_text}\n\n"

    await update.message.reply_text(message)


async def last_report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    report_path = LAST_REPORT_BY_CHAT.get(chat_id)

    if not report_path:
        reports = sorted(REPORTS_DIR.glob("report_*.json"), reverse=True)
        if reports:
            report_path = str(reports[0])

    if not report_path or not Path(report_path).exists():
        await update.message.reply_text("Aucun rapport JSON disponible.")
        return

    await update.message.reply_document(
        document=open(report_path, "rb"),
        filename=Path(report_path).name,
        caption="Voici le dernier rapport JSON."
    )


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Commande inconnue. Utilise /help pour voir les commandes disponibles."
    )


def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN est manquant dans .env")

    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("verify", verify_command))
    app.add_handler(CommandHandler("history", history))
    app.add_handler(CommandHandler("lastreport", last_report))

    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.COMMAND, unknown_command))

    print("Bot Telegram Nissan lancé en mode sérieux...")
    print("Commandes : /start /help /status /verify /history /lastreport")
    print("CTRL + C pour arrêter.")

    app.run_polling()


if __name__ == "__main__":
    main()