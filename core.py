import discord.ext.commands
import torch
import redis
import discord
from discord.ext import commands
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import discord.ext


PATH = 'khvatov/ru_toxicity_detector'
tokenizer = AutoTokenizer.from_pretrained(PATH)
model = AutoModelForSequenceClassification.from_pretrained(PATH)

model.to(torch.device("cpu"))


def rating_change(id: int, what: bool) -> None:
    DB_client = redis.Redis()
    get = DB_client.get(id)

    if get is None:
        DB_client.set(id, 0)
    else:
        if what:
            DB_client.set(id, int(get)+1)
        else:
            DB_client.set(id, int(get)-1)
    DB_client.close()

def rating_get(id: int) -> int:
    DB_client = redis.Redis()
    get = redis.Redis().get(id)

    if get is None:
        DB_client.set(id, 0)
        return 0
    else:
        return get


def get_toxicity_probs(text: str) -> list:
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
        proba = torch.nn.functional.softmax(model(**inputs).logits, dim=1).cpu().numpy()
    return proba[0]

async def analyse_message(message: discord.Message) -> None:
    result = get_toxicity_probs(message.content)
    if result[0] > result[1] and result[0] > 0.999:
        rating_change(message.author.id, True)

    if result[0] < result[1] and result[1] > 0.70:
        rating_change(message.author.id, False)


class Rating(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
    
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        await analyse_message(message)
    
    @commands.command(name="rating", aliases=["рейтинг"], pass_context = True)
    async def rating(self, ctx):
        await ctx.channel.send(f"Ваш социальный рейтинг: {int(rating_get(ctx.author.id))}")

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix = ".", intents=intents)

@bot.event
async def on_ready():
    await bot.add_cog(Rating(bot))
    print(f'We have logged in as {bot.user}')

bot.run("")