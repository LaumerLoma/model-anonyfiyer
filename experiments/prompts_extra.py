"""80 additional prompts for the v2 experiment (PROTOCOL_v2.md), same seven task types and counts as
prompts.py. Indices continue from 80. Do not edit after v2 data collection starts."""

from prompts import PROMPTS

EXTRA: list[tuple[str, str]] = [
    # explain
    ("explain", "How does a refrigerator keep food cold?"),
    ("explain", "Why do airplanes stay in the air?"),
    ("explain", "How does GPS determine your location?"),
    ("explain", "What makes a song catchy?"),
    ("explain", "How do credit scores work?"),
    ("explain", "Why does ice float on water?"),
    ("explain", "How does a search engine rank web pages?"),
    ("explain", "What is the greenhouse effect?"),
    ("explain", "How do muscles grow stronger after exercise?"),
    ("explain", "Why do we dream?"),
    ("explain", "How does an electric motor work?"),
    ("explain", "What is supply and demand?"),
    # advice
    ("advice", "How can I stop procrastinating?"),
    ("advice", "What should I pack for a week-long hiking trip?"),
    ("advice", "How do I make friends in a new city?"),
    ("advice", "How should I organize my personal finances?"),
    ("advice", "How can I become a better writer?"),
    ("advice", "What should I look for when buying a used car?"),
    ("advice", "How do I prepare for my first marathon?"),
    ("advice", "How can I reduce stress at work?"),
    ("advice", "How should I choose a university major?"),
    ("advice", "How do I keep houseplants alive?"),
    ("advice", "What is a good way to learn to cook?"),
    ("advice", "How can I spend less time on my phone?"),
    # coding
    ("coding", "Write a Python function that returns the n-th Fibonacci number."),
    ("coding", "Explain the difference between processes and threads."),
    ("coding", "What is the difference between an abstract class and an interface?"),
    ("coding", "Write a function that removes duplicates from a list while keeping order."),
    ("coding", "How does garbage collection work?"),
    ("coding", "Explain what recursion is with a simple example."),
    ("coding", "What is the difference between unit tests and integration tests?"),
    ("coding", "How do I read a CSV file in Python?"),
    ("coding", "What is a REST API?"),
    ("coding", "Explain the difference between stack and heap memory."),
    ("coding", "Write a binary search function."),
    ("coding", "What are environment variables and how are they used?"),
    # math
    ("math", "What is the area of a circle with a radius of 4 cm?"),
    ("math", "Explain what a logarithm is."),
    ("math", "What is the expected value of a fair six-sided die?"),
    ("math", "How do you convert a fraction to a percentage?"),
    ("math", "Explain the Pythagorean theorem with an example."),
    ("math", "What is a standard deviation?"),
    ("math", "If a shirt costs 40 after a 20% discount, what was the original price?"),
    ("math", "Explain what an integral is, intuitively."),
    ("math", "What is the sum of the angles in a hexagon?"),
    ("math", "How many different 4-digit PIN codes are possible?"),
    # creative
    ("creative", "Write a short poem about a city in the rain."),
    ("creative", "Write a fable about a fox and a crow with a modern moral."),
    ("creative", "Describe an alien marketplace."),
    ("creative", "Write a haiku about winter."),
    ("creative", "Write a product description for a time machine."),
    ("creative", "Write the first lines of a science fiction story set on Mars."),
    ("creative", "Describe the sound of a forest to someone who cannot hear."),
    ("creative", "Write a short letter from a lighthouse keeper."),
    ("creative", "Write a children's rhyme about brushing teeth."),
    ("creative", "Describe an old library from the point of view of a book."),
    # opinion
    ("opinion", "Should voting be mandatory?"),
    ("opinion", "Is a college degree still worth it?"),
    ("opinion", "What are the pros and cons of universal basic income?"),
    ("opinion", "Should artificial intelligence be regulated?"),
    ("opinion", "Is it better to specialize or be a generalist?"),
    ("opinion", "Should zoos exist?"),
    ("opinion", "Is tipping a good system?"),
    ("opinion", "Should children have smartphones?"),
    ("opinion", "What are the pros and cons of open-plan offices?"),
    ("opinion", "Is it better to read books on paper or on screens?"),
    ("opinion", "Should public transport be free?"),
    ("opinion", "Is it ethical to eat meat?"),
    # factual
    ("factual", "What caused the extinction of the dinosaurs?"),
    ("factual", "How does the human heart pump blood?"),
    ("factual", "What is the largest desert in the world?"),
    ("factual", "Who was Marie Curie?"),
    ("factual", "How do earthquakes happen?"),
    ("factual", "What was the Renaissance?"),
    ("factual", "How do vaccines get approved?"),
    ("factual", "What is the difference between an asteroid and a comet?"),
    ("factual", "How does the immune system fight infections?"),
    ("factual", "What was the Industrial Revolution?"),
    ("factual", "How do bees make honey?"),
    ("factual", "What is the theory of evolution?"),
]

assert len(EXTRA) == 80, len(EXTRA)
assert not {q for _, q in EXTRA} & {q for _, q in PROMPTS}, "overlap with prompts.py"

ALL_PROMPTS: list[tuple[str, str]] = PROMPTS + EXTRA
