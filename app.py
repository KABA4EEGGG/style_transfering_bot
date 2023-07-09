import logging
from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.dispatcher import FSMContext
import os
from sys import exit
from model import *
from transform_funcs import *
import gc

bot_token = os.getenv("BOT_TOKEN")
if not bot_token:
    exit("Error: no token provided")

# bot initialize
bot = Bot(token=bot_token)
dp = Dispatcher(bot, storage=MemoryStorage())
# simple logging
logging.basicConfig(level=logging.INFO)
# model initialize
style_model = Net(ngf=128)
style_model.load_state_dict(torch.load('pretrained.model'), False)

# Initializing flags to check for images.
content_flag = False
style_flag = False


# FSM initialize
class GetPictures(StatesGroup):
    wait_photo = State()
    wait_another_photo = State()


def transform(content_root, style_root, im_size):
    """Function for image transformation."""
    content_image = tensor_load_rgbimg(content_root, size=im_size,
                                       keep_asp=True).unsqueeze(0)
    style = tensor_load_rgbimg(style_root, size=im_size).unsqueeze(0)
    style = preprocess_batch(style)
    style_v = Variable(style)
    content_image = Variable(preprocess_batch(content_image))
    style_model.setTarget(style_v)
    output = style_model(content_image)
    tensor_save_bgrimg(output.data[0], 'result' + user_id + '.jpg', False)

    # Clear the RAM.
    del content_image
    del style
    del style_v
    del output
    torch.cuda.empty_cache()
    gc.collect()


# creating keyboards 
keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
buttons = ["style transfer", "help"]
keyboard.add(*buttons)
back_keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
back_keyboard.add("back")
style_keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
buttons_style = ["ok", "back"]
style_keyboard.add(*buttons_style)


@dp.message_handler(commands="start", state='*')
async def cmd_start(message: types.Message):
    global user_name
    user_name = str(message.from_user.first_name)
    await message.answer(text=f"Hello, *{user_name}*! A bot that can transfer\
    the style of photos is in touch. Let's try it?", reply_markup=keyboard, parse_mode="Markdown")


@dp.message_handler(lambda message: message.text == "help", state='*')
async def help_message(message: types.Message):
    """
    Instructions for using the bot
    """
    await message.answer(text="I want to tell you how to use this bot: \n"
                              "*1) Load photo with your content first*\n"
                              "*2) Then, load photo with style*\n"
                              "*3) Choose quality of result photo*\n"
                              "*4) Wait a little while and get a photo of the result*", parse_mode="Markdown")


@dp.message_handler(lambda message: message.text in ("style transfer", "back"), state='*')
@dp.message_handler(commands=['transfer'], state='*')
async def start_style_transfer(message: types.Message):
    global user_id
    user_id = str(message.from_user.id)
    logging.info(f"{user_id} STARTS STYLE TRANSFER")
    await message.answer(text="Please, send content image to me.", reply_markup=None)
    await GetPictures.wait_photo.set()


@dp.message_handler(state=GetPictures.wait_photo, content_types=types.ContentTypes.PHOTO)
async def photo_processing(message):
    """
    Triggered after sending a photo
    """
    global content_flag
    global style_flag
    # The bot is waiting for a picture with content from the user.
    if not content_flag:
        logging.info(f"{user_id} DOWNLOAD CONTENT IMAGE")
        await message.photo[-1].download('content' + user_id + '.jpg')
        await message.answer(text='I got the **content** image.\n'
                                  'Now send the **style** image.\n'
                                  'Or you can swap images (use the back button).', reply_markup=back_keyboard,
                             parse_mode='Markdown')

        content_flag = True  # Now the bot knows that the content image exists.

    # The bot is waiting for a picture with style from the user.
    else:
        logging.info(f"{user_id} DOWNLOAD STYLE IMAGE")
        await message.photo[-1].download('style' + user_id + '.jpg')
        await message.answer(text='I got the **style** image.\n'
                                  'If you are sure about the images you have sent, then click *ok*.'
                                  'Or you can swap images (use the *back* button).', reply_markup=style_keyboard,
                             parse_mode="Markdown")

        style_flag = True

    await GetPictures.wait_photo.set()


@dp.message_handler(lambda message: message.text == "ok",
                    state=GetPictures.wait_photo)
async def run_style_transfer(message: types.Message, state: FSMContext):
    """Preparing for image processing."""

    # Let's make sure that the user has added both images.
    logging.info(f"{user_id} ADDED IMAGES")
    if not (content_flag * style_flag):
        await message.answer(text="Upload both images please.")
        return

    # Adding answer options.
    res = types.ReplyKeyboardMarkup(resize_keyboard=True,
                                    one_time_keyboard=True)
    res_buttons = ["Bad", "Medium", "Best"]
    res.add(*res_buttons)
    await message.answer(text="Now you need to choose the quality of the resulting image"
                              "The better the quality of the image you choose, the longer it will take to process the "
                              "image.",
                         reply_markup=res)


@dp.message_handler(lambda message: message.text in ("Bad", "Medium", "Best"),
                    state=GetPictures.wait_photo)
async def processing(message: types.Message, state: FSMContext):
    """Image processing depending on the selected quality."""
    global content_flag
    global style_flag

    if message.text == 'Bad':
        image_size = 400
    elif message.text == 'Medium':
        image_size = 450
    else:
        image_size = 600
    await message.answer(text='Style transferring starts.\n'
                              'Wait some time...',
                         reply_markup=types.ReplyKeyboardRemove())
    transform('content' + user_id + '.jpg', 'style' + user_id + '.jpg', image_size)
    with open('result' + user_id + '.jpg', 'rb') as file:
        await message.answer_photo(file, caption='Work is done!', reply_markup=keyboard)
    content_flag = False
    style_flag = False
    os.remove('content' + user_id + '.jpg')
    os.remove('style' + user_id + '.jpg')
    os.remove('result' + user_id + '.jpg')
    await state.finish()


@dp.message_handler(state='*')
async def bad_commands(message: types.Message):
    await message.answer(text="Sorry, I don't know this command \n"
                              "Write *'/'* to see list of commands or press */help*",
                         reply_markup=keyboard, parse_mode='Markdown')


if __name__ == "__main__":
    # start bot
    executor.start_polling(dp, skip_updates=True)
