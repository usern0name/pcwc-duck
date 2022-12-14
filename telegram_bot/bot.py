import json
import logging
import numpy as np
import tensorflow as tf
import requests
from PIL import Image

from telegram import __version__ as TG_VER

try:
    from telegram import __version_info__
except ImportError:
    __version_info__ = (0, 0, 0, 0, 0)  # type: ignore[assignment]

if __version_info__ < (20, 0, 0, "alpha", 1):
    raise RuntimeError(
        f"This example is not compatible with your current PTB version {TG_VER}. To view the "
        f"{TG_VER} version of this example, "
        f"visit https://docs.python-telegram-bot.org/en/v{TG_VER}/examples.html"
    )
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    print("on start command")
    user = update.effective_user
    await update.message.reply_html(
        rf"{user.mention_html()}!",
        reply_markup=ForceReply(selective=True),
    )


async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    file_id = update.message.photo[-1].file_id
    file_model = requests.get(
        f'https://api.telegram.org/bot5984115984:AAGIriAnePRYSp_kKK98s-tDzdPY7Rcyyjs/getfile?file_id={file_id}',
        stream=True, verify=False)
    file_path = json.loads(file_model.text)['result']['file_path']

    response = requests.get(
        f'https://api.telegram.org/file/bot5984115984:AAGIriAnePRYSp_kKK98s-tDzdPY7Rcyyjs/{file_path}', stream=True,
        verify=False)
    image = Image.open(response.raw)

    new_image = image.resize((224, 224))
    image_np = np.array(new_image).astype(np.uint8)
    imagenet_labels = np.array(open('Loc_ImageNetLabels.txt', encoding='utf-8').read().splitlines())
    image_np = tf.keras.applications.mobilenet.preprocess_input(
        image_np[tf.newaxis, ...])

    pretrained_model = tf.keras.models.load_model('C:\\Empty')
    result_before_save = pretrained_model(image_np)
    decoded = imagenet_labels[np.argsort(result_before_save)[0, ::-1][:5] + 1]

    human_result = ''

    for element in decoded:
        human_result += element + '\n'

    await update.message.reply_text(human_result)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print("on help command")
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Help!")


async def detect_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('update.message.text')


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    await update.message.reply_text(update.message.text)

async def anek_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    response = requests.get(
        'http://rzhunemogu.ru/RandJSON.aspx?CType=1',
        stream=False, verify=False)
    model = json.loads(response.text.replace("\r\n", "xNx"))
    await update.message.reply_text(model['content'].replace("xNx", "\n"))

def init_bot() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token("5984115984:AAGIriAnePRYSp_kKK98s-tDzdPY7Rcyyjs").build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("detect", detect_command))
    application.add_handler(CommandHandler("anek", anek_command))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
    application.add_handler(MessageHandler(filters.PHOTO, photo))
    # Run the bot until the user presses Ctrl-C
    application.run_polling()
