"""
bot.py  —  TikTok Carousel Generator Telegram Bot
Deploy on Render / Railway / any server with webhook or polling.

Message format:
    /carousel
    Topic: 10 story games you must play
    Games:
    Elden Ring 2022
    Hollow Knight 2017
    Disco Elysium 2019

Or compact (year optional, defaults to current year):
    /carousel
    Topic: Best RPGs ever
    Games:
    Baldur's Gate 3
    Witcher 3
    Elden Ring 2022
"""

import os
import re
import asyncio
import logging
from datetime import datetime

import random

from aiogram import Bot, Dispatcher, F
from aiogram.types import (
    Message, BufferedInputFile, InputMediaPhoto,
)
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage

from carousel_generator import generate_carousel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

import os
from dotenv import load_dotenv

# Загружаем переменные из файла .env в окружение
load_dotenv()

# Теперь os.environ сможет их прочитать как на твоем ПК, так и на сервере
BOT_TOKEN = os.getenv("BOT_TOKEN")

bot = Bot(token=BOT_TOKEN)
dp  = Dispatcher(storage=MemoryStorage())



import json

def get_idea():
    # 1. Читаем данные из файла ideas.json
    try:
        with open('ideas.json', 'r', encoding='utf-8') as f:
            carousels = json.load(f)
    except FileNotFoundError:
        logger.error("Файл ideas.json не найден!")
        return "⚠️ Ошибка: Файл с идеями (ideas.json) не найден на сервере."
    except json.JSONDecodeError:
        logger.error("ideas.json содержит некорректный формат JSON.")
        return "⚠️ Ошибка: Файл ideas.json поврежден (некорректный JSON)."

    # Если список пуст
    if not carousels:
        return "📭 Все идеи закончились! Файл ideas.json пуст."

    # 2. Выбираем случайную карусель
    selected_item = random.choice(carousels)

    # 3. Удаляем выбранную карусель из общего списка
    carousels.remove(selected_item)

    # 4. Перезаписываем файл ideas.json уже БЕЗ этой идеи
    try:
        with open('ideas.json', 'w', encoding='utf-8') as f:
            json.dump(carousels, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception("Не удалось обновить файл ideas.json")
        # Не блокируем работу, но предупреждаем в логах

    # 5. Форматируем результат для отправки в Telegram
    output_text = "/carousel\n"
    output_text += f"Topic: {selected_item['topic']}\n"
    output_text += "Games:\n"

    for game in selected_item['games']:
        output_text += f"{game['name']} {game['year']}\n"

    return output_text

# ── PARSER ─────────────────────────────────────────────────────────────────────

def parse_carousel_request(text: str):
    """
    Parses the message body (without /carousel command).

    Returns (topic: str, games: list[dict]) or raises ValueError.

    Accepted formats:
        Topic: <topic text>
        Games:
        <Game Name> [year]
        ...
    or without "Topic:" / "Games:" headers if user sends it inline.
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]

    topic = None
    games_raw: list[str] = []
    in_games = False

    for line in lines:
        low = line.lower()
        if low.startswith("topic:"):
            topic = line[6:].strip()
            in_games = False
        elif low.startswith("games:"):
            in_games = True
        elif in_games:
            games_raw.append(line)
        elif topic is None and not in_games:
            # First non-empty line with no headers → treat as topic
            topic = line

    if not topic:
        raise ValueError(
            "Could not find a topic.\n"
            "Use format:\n<b>Topic:</b> 10 story games\n<b>Games:</b>\nElden Ring 2022\n..."
        )

    if not games_raw:
        raise ValueError(
            "Could not find a games list.\n"
            "After <b>Games:</b> list one game per line, optionally followed by a year."
        )

    current_year = str(datetime.now().year)
    games = []
    for raw in games_raw:
        # Try to extract trailing 4-digit year
        m = re.search(r"\b(19|20)\d{2}\b", raw)
        if m:
            year = m.group(0)
            name = raw[:m.start()].strip().strip("-—").strip()
        else:
            year = current_year
            name = raw.strip()
        if name:
            games.append({"name": name, "year": year})

    if not games:
        raise ValueError("Games list is empty after parsing.")

    return topic, games


# ── HANDLER ────────────────────────────────────────────────────────────────────

HELP_TEXT = (
    "🎮 <b>TikTok Carousel Generator</b>\n\n"
    "Send me a /carousel command followed by:\n\n"
    "<b>Topic:</b> 10 story games you must play\n"
    "<b>Games:</b>\n"
    "Elden Ring 2022\n"
    "Hollow Knight 2017\n"
    "Disco Elysium 2019\n"
    "Baldur's Gate 3 2023\n\n"
    "I'll generate a full TikTok carousel for you!\n"
    "Year is optional — defaults to the current year.\n\n"
    "⚠️ Processing takes ~1 min depending on how many games you list."
)


@dp.message(Command("start", "help"))
async def cmd_start(msg: Message):
    await msg.answer(HELP_TEXT, parse_mode="HTML")


@dp.message(Command("idea", "help"))
async def cmd_idea(msg: Message):
    await msg.answer(get_idea(), parse_mode="HTML")


@dp.message(Command("carousel"))
async def cmd_carousel(msg: Message):
    # Strip the /carousel command from the text
    text = msg.text or ""
    body = re.sub(r"^/carousel\S*\s*", "", text, count=1, flags=re.IGNORECASE).strip()

    if not body:
        await msg.answer(
            "Please include your request after /carousel.\n\n" + HELP_TEXT,
            parse_mode="HTML",
        )
        return

    try:
        topic, games = parse_carousel_request(body)
    except ValueError as e:
        await msg.answer(f"⚠️ {e}", parse_mode="HTML")
        return

    status = await msg.answer(
        f"⏳ Generating carousel for <b>{topic}</b>\n"
        f"  {len(games)} game(s) in your list…\n\n"
        "This may take a minute, please wait.",
        parse_mode="HTML",
    )

    try:
        # Run CPU-heavy generation in a thread pool so the event loop isn't blocked
        loop = asyncio.get_event_loop()
        slides: list[bytes] = await loop.run_in_executor(
            None, generate_carousel, topic, games
        )
    except ValueError as e:
        await status.edit_text(f"❌ {e}")
        return
    except Exception as e:
        logger.exception("Carousel generation failed")
        await status.edit_text(f"❌ Unexpected error: {e}")
        return

    await status.edit_text(
        f"✅ Done! Sending {len(slides)} slides…"
    )

    # Label mapping for file names
    def slide_label(idx, total):
        if idx == 0:
            return "intro"
        if idx == total - 1:
            return "outro"
        return f"slide_{idx:02d}"

    total = len(slides)

    # Telegram allows max 10 media per album
    CHUNK = 10
    for chunk_start in range(0, total, CHUNK):
        chunk = slides[chunk_start: chunk_start + CHUNK]
        media_group = [
            InputMediaPhoto(
                media=BufferedInputFile(
                    data,
                    filename=f"{slide_label(chunk_start + i, total)}.jpg",
                ),
                caption=(
                    f"🎬 <b>{topic}</b>\n#{chunk_start + i + 1}/{total}"
                    if i == 0 and chunk_start == 0
                    else None
                ),
                parse_mode="HTML" if (i == 0 and chunk_start == 0) else None,
            )
            for i, data in enumerate(chunk)
        ]
        await msg.answer_media_group(media=media_group)
        await asyncio.sleep(0.5)  # avoid flood limits

    await msg.answer(
        f"🎉 Carousel ready!\n"
        f"<b>{total}</b> slides for: <i>{topic}</i>",
        parse_mode="HTML",
    )


@dp.message(F.text)
async def fallback(msg: Message):
    await msg.answer(
        "Use /carousel to generate a carousel, or /help for instructions.",
        parse_mode="HTML",
    )


# ── ENTRY POINT ────────────────────────────────────────────────────────────────

async def main():
    logger.info("Bot starting (polling mode)…")
    await dp.start_polling(bot, skip_updates=True)


if __name__ == "__main__":
    asyncio.run(main())
