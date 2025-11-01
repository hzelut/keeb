#!/usr/bin/env python3

import urllib.request
import urllib.error
import socket
import shutil
import zipfile
import tempfile
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import os
import re
import math
import random
import multiprocessing
import copy
from collections import Counter
import traceback
from ast import literal_eval
import signal
from dataclasses import dataclass, field, fields
import statistics
import sys
from itertools import permutations

@dataclass
class Score:
	effort: int = 0
	sfb: int = 0
	rollin: int = 0
	scissors: int = 0

@dataclass
class Layout:
	letters: list[list[str]]
	score: Score = field(default_factory=lambda: Score())
	total: int = 0
	left_usage: int = 0
	right_usage: int = 0

	def clone(self):
		return Layout([row[:] for row in self.letters])

	def __post_init__(self):
		self.letters = copy.deepcopy(self.letters)
		self.calc_scores()
		if SCORE_MEDIAN is not None:
			self.calc_total_score()

	def __eq__(self, other):
		if not isinstance(other, Layout):
			return False
		return self.letters == other.letters

	def __hash__(self):
		return hash(tuple(tuple(r) for r in self.letters))

	def calc_scores(self):
		self.score = Score()
		self.left_usage = 0
		self.right_usage = 0
		pos = {}
		max_e = max(max(r) for r in EFFORT_GRID)

		for r in range(3):
			for c in range(10):
				ch = self.letters[r][c]
				if ch != ' ':
					pos[ch] = (r, c)
					try:
						l = LETTERS[ch]
					except KeyError:
						print('======= ERROR')
						print_layout(self)
						sys.exit(1)
					self.score.effort += l * EFFORT_GRID[r][c]
					if c < 5:
						self.left_usage += l
					else:
						self.right_usage += l

		for pair, count in BIGRAMS.items():
			a, b = pair[0], pair[1]
			if a not in pos or b not in pos:
				continue
			r1, c1 = pos[a]
			r2, c2 = pos[b]
			f1 = FINGER_GRID[r1][c1]
			f2 = FINGER_GRID[r2][c2]
			e1 = EFFORT_GRID[r1][c1]
			e2 = EFFORT_GRID[r2][c2]
			if (1 <= f1 <= 4 and 1 <= f2 <= 4) or (5 <= f1 <= 8 and 5 <= f2 <= 8):
				if f1 == f2:
					self.score.sfb += count * (e1+e2)
				elif abs(f1 - f2) == 1:
					if f2 < f1 and abs(r1 - r2) <= 1:
						self.score.rollin += count * (max_e*2 - (e1+e2))
					elif abs(r1 - r2) == 2:
						weight = 1.0
						if (f1 < f2 and r1 == 0) or (f2 < f1 and r2 == 0):
							weight = 1.3
						self.score.scissors += count * (e1+e2) * weight

	def calc_total_score(self):
		def norm(v, m, d):
			return (v - m) / d if d != 0 else 0.0

		r = Score(
			effort=norm(self.score.effort, SCORE_MEDIAN.effort, SCORE_IQR.effort),
			sfb=norm(self.score.sfb, SCORE_MEDIAN.sfb, SCORE_IQR.sfb),
			rollin=norm(self.score.rollin, SCORE_MEDIAN.rollin, SCORE_IQR.rollin),
			scissors=norm(self.score.scissors, SCORE_MEDIAN.scissors, SCORE_IQR.scissors),
		)

		self.total = (
			(-r.effort) * SCORE_RATES.effort +
			(-r.sfb) * SCORE_RATES.sfb +
			(r.rollin) * SCORE_RATES.rollin +
			(-r.scissors) * SCORE_RATES.scissors
		)

TMP_PATH = None
ANALYZE_RESULT_FILENAME = 'analyze_result.tsv'
BEST_RESULT_FILENAME = 'best.txt'
BEST_SIZE = 5

LETTERS = Counter()
BIGRAMS = Counter()
SYMBOLS = Counter()
SYMBOL_BIGRAMS = Counter()

EFFORT_GRID = [
	[5.0, 3.0, 2.4, 2.0, 2.2, 2.2, 2.0, 2.4, 3.0, 5.0],
	[9.0, 1.5, 1.3, 1.1, 1.0, 1.0, 1.1, 1.3, 1.5, 9.0],
	[9.0, 3.2, 2.6, 2.3, 1.6, 1.6, 2.3, 2.6, 3.2, 9.0],
]

FINGER_GRID = [
	[4, 4, 3, 2, 1, 5, 6, 7, 8, 8],
	[4, 4, 3, 2, 1, 5, 6, 7, 8, 8],
	[4, 4, 3, 2, 1, 5, 6, 7, 8, 8],
]

SCORE_RATES = Score(
	effort = 0.3,
	sfb = 0.2,
	rollin = 0.1,
	scissors = 0.4,
)
SCORE_MEDIAN = None
SCORE_IQR = None

def layout_key(l):
	return (
		l.total,
		-abs(l.left_usage - l.right_usage),
		-l.score.scissors,
		-l.score.effort,
		-l.score.sfb,
		l.score.rollin,
	)

def best_layout(layouts: list[Layout]):
	return max(layouts, key=layout_key).clone()

def sort_layouts(layouts: list[Layout]):
	layouts.sort(key=layout_key, reverse=True)
	return layouts

def sort_unique_layouts(layouts: list[Layout], size=BEST_SIZE):
	layouts = sort_layouts(list(set(layouts)))
	result = []

	for layout in layouts:
		if len(result) == size:
			break
		if any(ch in layout.letters[1][:4] for ch in 'cyd'):
			continue
		if any(layout.letters[1][:4] == o.letters[1][:4] for o in result):
			continue
		result.append(layout)

	return result

def init_score_state():
	global SCORE_MEDIAN, SCORE_IQR 
	def iqr(v):
		q = statistics.quantiles(v, n=4)
		return q[2] - q[0]

	base_layout = make_initial_layout()
	unique_layouts = {base_layout.clone()}
	while len(unique_layouts) < 100:
		unique_layouts.add(make_random(base_layout))
	layouts = list(unique_layouts)

	vals = {f.name: [getattr(l.score, f.name) for l in layouts] for f in fields(Score)}

	SCORE_MEDIAN = Score(**{k: statistics.median(v) for k, v in vals.items()})

	iqr_map = {}
	for k, v in vals.items():
		d = iqr(v)
		iqr_map[k] = d if d != 0 else 1e-9
	SCORE_IQR = Score(**iqr_map)

def check_target_url(url):
	try:
		req = urllib.request.Request(url, method='HEAD')
		with urllib.request.urlopen(req, timeout=5) as res:
			if res.status == 200: return True
	except (urllib.error.HTTPError, urllib.error.URLError, socket.timeout): pass

	return False

def download_target(url, dest):
	repo_name = url.rstrip('/').split('/')[-1]
	base_url = url.rstrip('/') + '/archive/refs/heads/'

	for u in [base_url+'master.zip', base_url+'main.zip']:
		try:
			with urllib.request.urlopen(u) as res:
				z = os.path.join(dest, f'{repo_name}.zip')
				with open(z, 'wb') as f:
					while True:
						buf = res.read(8192)
						if not buf:
							break
						f.write(buf)

			with zipfile.ZipFile(z, 'r') as zz:
				zz.extractall(dest)
			os.remove(z)
			return True
		except (urllib.error.HTTPError, urllib.error.URLError, socket.timeout):
			try: os.remove(z)
			except: pass

	return False

def cleanup(sig, frame):
	global TMP_PATH
	try:
		if TMP_PATH and os.path.exists(TMP_PATH):
			shutil.rmtree(TMP_PATH)
	except Exception:
		pass
	sys.exit(1)

def save_analyze_result(result_path):
	file_path = os.path.join(result_path, ANALYZE_RESULT_FILENAME)

	with open(file_path, 'w', encoding='utf-8') as f:
		f.write('letter\tfrequency\n')
		for ch, count in LETTERS.most_common():
			f.write(f'{ch}\t{count}\n')

		f.write('\nbigram\tfrequency\n')
		for bg, count in BIGRAMS.most_common():
			f.write(f'{bg}\t{count}\n')

		f.write('\nsymbol\tfrequency\n')
		for ch, count in SYMBOLS.most_common():
			f.write(f'{ch}\t{count}\n')

		f.write('\nsymbigram\tfrequency\n')
		for bg, count in SYMBOL_BIGRAMS.most_common():
			f.write(f'{bg}\t{count}\n')

def load_analysis_result(result_path):
	global LETTERS, BIGRAMS, SYMBOLS, SYMBOL_BIGRAMS

	file_path = os.path.join(result_path, ANALYZE_RESULT_FILENAME)

	with open(file_path, 'r', encoding='utf-8') as f:
		section = None
		for line in f:
			line = line.strip()
			if not line:
				continue
			if line.startswith('letter\t'):
				section = 'letters'
				continue
			elif line.startswith('bigram\t'):
				section = 'bigrams'
				continue
			elif line.startswith('symbol\t'):
				section = 'symbols'
				continue
			elif line.startswith('symbigram\t'):
				section = 'symbigrams'
				continue

			if section == 'letters':
				ch, count = line.split('\t')
				LETTERS[ch] = int(count)
			elif section == 'bigrams':
				bg, count = line.split('\t')
				BIGRAMS[bg] = int(count)
			elif section == 'symbols':
				ch, count = line.split('\t')
				SYMBOLS[ch] = int(count)
			elif section == 'symbigrams':
				bg, count = line.split('\t')
				SYMBOL_BIGRAMS[bg] = int(count)

def analyze_target_single(full_path):
	letters = Counter()
	bigrams = Counter()
	symbols = Counter()
	symbol_bigrams = Counter()
	pattern = re.compile('[a-z]+', re.IGNORECASE)
	symbol_set = set('~`!@#$%^&*()-_=+[]{}\\|;:\'",.<>/?')
	try:
		with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
			text = f.read()
			groups = pattern.findall(text)
			for g in groups:
				word = g.lower()
				word = [ch for ch in word if 'a' <= ch <= 'z']
				for ch in word:
					letters[ch] += 1
				for i in range(len(word)-1):
					if word[i] != word[i+1]:
						bigrams[word[i] + word[i+1]] += 1
			for ch in text:
				if ch in symbol_set:
					symbols[ch] += 1
			for i in range(len(text) - 1):
				if text[i] in symbol_set and text[i + 1] in symbol_set and text[i] != text[i + 1]:
					symbol_bigrams[text[i] + text[i + 1]] += 1
	except (FileNotFoundError, PermissionError, IsADirectoryError) as e:
		print(f'Failed: {full_path} — {e}')

	return letters, bigrams, symbols, symbol_bigrams

def analyze_target(result_path):
	global LETTERS, BIGRAMS, SYMBOLS, SYMBOL_BIGRAMS, TMP_PATH

	targets = [
		'https://github.com/torvalds/linux',            # C
		'https://github.com/opencv/opencv',             # C++
		'https://github.com/gcc-mirror/gcc',            # C++
		'https://github.com/llvm/llvm-project',         # C++/C
		'https://github.com/python/cpython',            # C/Python
		'https://github.com/numpy/numpy',               # Python/C
		'https://github.com/django/django',             # Python
		'https://github.com/psf/requests',              # Python
		'https://github.com/facebook/react',            # JavaScript/TypeScript
		'https://github.com/microsoft/vscode',          # TypeScript
		'https://github.com/sveltejs/svelte',           # JavaScript/TypeScript
		'https://github.com/nodejs/node',               # JavaScript/C++
		'https://github.com/denoland/deno',             # TypeScript/Rust
		'https://github.com/kubernetes/kubernetes',     # Go
		'https://github.com/golang/go',                 # Go
		'https://github.com/rust-lang/rust',            # Rust
		'https://github.com/theseus-os/Theseus',        # Rust
		'https://github.com/bytecodealliance/wasmtime', # Rust
		'https://github.com/sharkdp/fd',                # Rust
		'https://github.com/ziglang/zig',               # Zig
		'https://github.com/vlang/v',                   # V
		'https://github.com/nim-lang/Nim',              # Nim
		'https://github.com/carbon-language/carbon-lang', # Carbon
		'https://github.com/ValeLang/Vale',             # Vale
		'https://github.com/rails/rails',               # Ruby
		'https://github.com/elixir-lang/elixir',        # Elixir
		'https://github.com/apple/swift',               # Swift
		'https://github.com/JetBrains/kotlin',          # Kotlin
		'https://github.com/php/php-src',               # PHP
		'https://github.com/lua/lua',                   # Lua
		'https://github.com/ghc/ghc',                   # Haskell
		'https://github.com/scala/scala',               # Scala
		'https://github.com/wch/r-source',              # R
		'https://github.com/dotnet/runtime',            # C#
		'https://github.com/java/jdk',                  # Java
		'https://github.com/ohmyzsh/ohmyzsh',           # Shell
		'https://github.com/copy/v86',                  # Assembly/JavaScript
		'https://github.com/cirosantilli/x86-bare-metal-examples', # Assembly/C
		'https://github.com/mit-pdos/xv6-public',       # C/Assembly
		'https://github.com/redox-os/redox',            # Rust/Assembly
		'https://github.com/SerenityOS/serenity',       # C++/Assembly
		'https://github.com/u-boot/u-boot',             # C/Assembly
		'https://github.com/coreboot/coreboot',         # C/Assembly
		'https://github.com/Maratyszcza/PeachPy',       # Python/Assembly
		'https://github.com/netwide-assembler/nasm',    # Assembly
		'https://github.com/BeaEngine/BeaEngine',       # Assembly/C
		'https://github.com/ApolloTeam-dev/AROS',       # C/Assembly
	]

	extensions = [
		'.c', '.h',
		'.cpp', '.hpp', '.cc', '.hh',
		'.ino',
		'.cs',
		'.java',
		'.kt', '.kts',
		'.scala',
		'.groovy',
		'.swift',
		'.m', '.mm',
		'.py',
		'.rb',
		'.js', '.jsx',
		'.ts', '.tsx',
		'.go',
		'.rs',
		'.zig',
		'.hs',
		'.ml', '.mli',
		'.ex', '.exs',
		'.erl',
		'.sh', '.bash', '.zsh',
		'.html', '.htm',
		'.css', '.scss',
		'.md', '.markdown',
		'.json', '.yaml', '.yml', '.toml',
		'.sql', '.proto', '.xml',
		'.dockerfile', 'Dockerfile',
		'.R', '.r',
		'.jl',
		'.php',
		'.pl', '.pm',
		'.lua',
		'.asm', '.s', '.S',
		'.v', '.sv', '.vhd', '.vhdl',
	]

	# Check
	print('[Check Target]')
	len_targets = len(targets)
	for i, url in enumerate(targets, 1):
		if check_target_url(url):
			print(f'\r\033[K{i}/{len_targets} ({i/len_targets*100:.1f}%) {url}', end='')

		else:
			print(f'\n[FAIL] {url}')
	print(f'\r\033[K...Done')

	# Download
	TMP_PATH = tempfile.mkdtemp(dir=os.path.expanduser('~'), prefix='keeb')
	print('[Download Target]')
	downloaded = 0
	with ThreadPoolExecutor(max_workers=8) as executor:
		futures = [executor.submit(download_target, url, TMP_PATH) for url in targets]
		for future in as_completed(futures):
			future.result()
			downloaded += 1
			print(f'\r\033[K{downloaded}/{len_targets} ({downloaded/len_targets*100:.1f}%)', end='')
	print(f'\r\033[K...Done')

	# file list
	files = []
	for root, dirs, fs in os.walk(TMP_PATH):
		for file in fs:
			_, ext = os.path.splitext(file)
			if ext in extensions:
				files.append(os.path.join(root, file))

	# Calc LETTERS, BIGRAMS
	print('[Analyze Target]')
	letters = Counter()
	bigrams = Counter()
	symbols = Counter()
	symbol_bigrams = Counter()
	len_files = len(files)
	with ProcessPoolExecutor() as executor:
		for i, (l, b, s, sb) in enumerate(executor.map(analyze_target_single, files), 1):
			letters += l
			bigrams += b
			symbols += s
			symbol_bigrams += sb
			print(f'\r\033[K{i}/{len_files} ({i/len_files*100:.1f}%)', end='')

	LETTERS = letters
	SYMBOLS = symbols
	total_count = sum(bigrams.values())
	threshold = total_count * 0.9
	cumulative = 0
	for bigram, count in bigrams.most_common():
		cumulative += count
		BIGRAMS[bigram] = count
		if cumulative >= threshold:
			break

	total_count = sum(symbol_bigrams.values())
	threshold = total_count * 0.9
	cumulative = 0
	for bigram, count in symbol_bigrams.most_common():
		cumulative += count
		SYMBOL_BIGRAMS[bigram] = count
		if cumulative >= threshold:
			break

	shutil.rmtree(TMP_PATH)
	print(f'\r\033[K...Done')

	# Store result
	save_analyze_result(result_path)

def make_initial_layout() -> Layout:
	grid = [] 
	coords = []
	for r in range(3):
		for c in range(10):
			coords.append((EFFORT_GRID[r][c], r, c))
	coords.sort()

	letters_sorted = [ch for ch, _ in LETTERS.most_common()]
	layout = [[' ' for _ in range(10)] for _ in range(3)]
	for i, (_, r, c) in enumerate(coords):
		if i < len(letters_sorted):
			layout[r][c] = letters_sorted[i]

	return Layout(layout)

def make_random(base_layout: Layout) -> Layout:
	layout = base_layout.clone()

	positions = [(i, j) for i in range(len(layout.letters)) for j in range(len(layout.letters[0])) if layout.letters[i][j] != ' ']
	keys = [layout.letters[i][j] for i, j in positions]
	random.shuffle(keys)

	for idx, (i, j) in enumerate(positions):
		layout.letters[i][j] = keys[idx]

	return layout

def crossover(parents: list[Layout], blank=' '):
	def flatten(layout):
		return [item for row in layout for item in row]
	def unflatten(flat, rows=3, cols=10):
		return [flat[i*cols:(i+1)*cols] for i in range(rows)]

	parent1 = flatten(parents[0].letters)
	parent2 = flatten(parents[1].letters)
	length = len(parent1)

	a, b = sorted(random.sample(range(length), 2))
	child = [blank if k == blank else None for k in parent1]
	child[a:b] = parent1[a:b]

	used = {k for k in child if k is not None}
	remain_keys = [k for k in parent2 if k not in used]
	it = iter(remain_keys)
	for i in range(length):
		if child[i] is None:
			child[i] = next(it)

	return Layout(unflatten(child))

def fine_tune_effort(base_layout: Layout):
	letters = [row[:] for row in base_layout.letters]
	positions = [(r,c) for r in range(3) for c in range(10) if letters[r][c] != ' ']
	positions.sort(key=lambda pos: LETTERS.get(letters[pos[0]][pos[1]],0), reverse=True)
	candidates = [base_layout.clone()]

	for (r,c) in positions:
		l = [row[:] for row in base_layout.letters]
		best = (r,c)
		for dr in (-1,0,1):
			for dc in (-1,0,1):
				nr,nc=r+dr,c+dc
				if 0<=nr<3 and 0<=nc<10 and l[nr][nc]!=' ':
					if EFFORT_GRID[nr][nc] < EFFORT_GRID[best[0]][best[1]]:
						best = (nr,nc)
		l[r][c], l[best[0]][best[1]] = l[best[0]][best[1]], l[r][c]
		candidates.append(Layout(l))

	return best_layout(candidates)

def optimize_effort(base_layout: Layout):
	orders = ['effort_asc', 'effort_desc', 'count_asc', 'count_desc']
	letters = [row[:] for row in base_layout.letters]
	layouts = {base_layout.clone()}

	for order in orders:
		effort_levels = list({val for row in EFFORT_GRID for val in row if val < 10})

		if order == 'effort_asc':
			effort_levels.sort()
		elif order == 'effort_desc':
			effort_levels.sort(reverse=True)
		else:
			effort_counts = {val: sum(1 for r in range(3) for c in range(10) if EFFORT_GRID[r][c] == val) for val in effort_levels}
			if order == 'count_asc':
				effort_levels.sort(key=lambda x: effort_counts[x])
			elif order == 'count_desc':
				effort_levels.sort(key=lambda x: -effort_counts[x])

		for effort_level in effort_levels:
			group_coords = [(r, c) for r in range(3) for c in range(10) if abs(EFFORT_GRID[r][c] - effort_level) <= 0.1]
			random.shuffle(group_coords)

			for i in range(len(group_coords)):
				r1, c1 = group_coords[i]
				for j in range(i+1, len(group_coords)):
					r2, c2 = group_coords[j]
					if letters[r1][c1] == ' ' and letters[r2][c2] == ' ':
						continue
					l = [row[:] for row in letters]
					l[r1][c1], l[r2][c2] = l[r2][c2], l[r1][c1]
					layouts.add(Layout(l))
	
	return best_layout(list(layouts))

def optimize_swap(base_layout: Layout, temperature, fix=0):
	n = None
	if fix == 0:
		if temperature > 50:
			n = random.choices([4, 5, 6], weights=[0.5, 0.3, 0.2], k=1)[0]
		elif temperature > 10:
			n = random.choices([3, 4], weights=[0.7, 0.3], k=1)[0]
		else:
			n = random.choices([2, 3], weights=[0.8, 0.2], k=1)[0]
	else:
		n = fix

	coords = set()
	while len(coords) < n:
		r, c = random.randint(0, 2), random.randint(0, 9)
		if base_layout.letters[r][c] != ' ':
			coords.add((r, c))
	coords = list(coords)

	letters = [row[:] for row in base_layout.letters]

	shuffled = coords[:]
	random.shuffle(shuffled)

	for i in range(n):
		r1, c1 = coords[i]
		r2, c2 = shuffled[i]
		letters[r1][c1], letters[r2][c2] = letters[r2][c2], letters[r1][c1]

	return Layout(letters)

def optimize_all_swap(base_layout: Layout):
	layouts = []
	l = [row[:] for row in base_layout.letters]

	letters = random.sample(list(LETTERS.keys()), 7)
	positions = [(r, c) for r in range(3) for c in range(10) if base_layout.letters[r][c] in letters]
	perms = permutations(letters, len(letters))

	for perm in perms:
		for (r, c), ch in zip(positions, perm):
			l[r][c] = ch
		layouts.append(Layout(l))

	layouts = sort_layouts(layouts)
	return best_layout(layouts)

def optimize_sa(base_layout: Layout, max_iter=10000, initial_temp=50.0, cooling_rate=0.9985, max_unimproved=2000):
	best = base_layout.clone()
	cur = base_layout.clone()
	temperature = initial_temp
	unimproved = 0

	for i in range(max_iter):
		new_layout = optimize_swap(cur, temperature)

		diff = new_layout.total - cur.total
		T = max(temperature, 1e-6)
		try:
			prob = math.exp(diff / T)
		except (OverflowError, ZeroDivisionError, TypeError):
			prob = 1.0

		unimproved += 1
		if diff >= 0 or prob > random.random():
			cur = new_layout
			if cur.total > best.total:
				best = cur.clone()
				temperature *= 1.05
				unimproved = 0
		temperature *= cooling_rate

		if unimproved > max_unimproved or temperature < 1e-3:
			break

	return best

def optimize(base_layouts: list[Layout], max_generation=10, max_population=100):
	best = [l.clone() for l in base_layouts[:BEST_SIZE]]

	# Init population
	unique_population = {best[0].clone()}
	while len(unique_population) < max_population:
		unique_population.add(make_random(best[0]))
	population = sort_layouts(list(unique_population))
	
	elites_len = max(1, int(max_population * 0.05))
	with ProcessPoolExecutor() as executor:
		for gen in range(1, max_generation+1):
			print(f'\r\033[K...{gen}/{max_generation}', end='')
			random_len = int(max_population* max(0.05, 0.3 * (1 - gen/ max_generation)))

			elites = [l.clone() for l in population[:elites_len]]
			parents = [best_layout(random.sample(population, 3)) for _ in range(max_population)]
			children = [crossover(random.sample(parents, 2)) for _ in range(max_population - elites_len - random_len)]

			# Make next
			population = []
			for elite in elites:
				population.append(fine_tune_effort(elite))
			for _ in range(random_len):
				population.append(make_random(best[0]))
			progress = min(gen/max_generation, 1.0)
			results = list(executor.map(
				optimize_worker,
				children,
				[progress] * len(children)
			))
			population.extend(results)
			population = sort_layouts(population)

			cur = best_layout(population)
			best.extend([l.clone() for l in population[:BEST_SIZE*2]])
			best = sort_layouts(list(set(best)))

	return best[:BEST_SIZE*2]

def optimize_worker(layout: Layout, progress):
	sa_weight = 0.2 + 0.2 * progress   # 0.2 - 0.4
	effort_weight = 0.2 + 0.1 * progress  # 0.2 - 0.3
	swap_weight = 0.3 - 0.05 * progress  # 0.3 - 0.25
	pass_weight = 1.0 - (sa_weight + effort_weight + swap_weight)

	weights = [max(0.0, sa_weight), max(0.0, effort_weight), max(0.0, swap_weight), max(0.0, pass_weight)]
	total = sum(weights)
	if total <= 0:
		weights = [0.25] * 4
		total = 1.0
	thresholds = []
	acc = 0.0
	for w in weights:
		acc += w / total
		thresholds.append(acc)

	r = random.random()
	if r < thresholds[0]:
		return optimize_sa(layout)
	elif r < thresholds[1]:
		return optimize_effort(layout)
	elif r < thresholds[2]:
		return optimize_all_swap(layout)
	else:
		return layout

def print_layout(layout: Layout):
	print(f'{layout.score.effort:,.0f}\t', end='')
	print(f'{layout.score.sfb:,.0f}\t', end='')
	print(f'{layout.score.rollin:,.0f}\t', end='')
	print(f'{layout.score.scissors:,.0f}')
	if layout.left_usage > 0:
		total = layout.left_usage + layout.right_usage
		left_percent = (layout.left_usage / total) * 100
		right_percent = (layout.right_usage / total) * 100
		print(f'{left_percent:.2f} : {right_percent:.2f} \t {layout.total:,}')
	for row in layout.letters:
		print(row)

def save_best_result(best, result_path):
	file_path = os.path.join(result_path, BEST_RESULT_FILENAME)
	with open(file_path, 'w', encoding='utf-8') as f:
		for l in best:
			for row in l.letters:
				print(row, file=f)

def load_best_result(result_path):
	layouts = []
	file_path = os.path.join(result_path, BEST_RESULT_FILENAME)
	with open(file_path, 'r', encoding='utf-8') as f:
		lines = [line.strip() for line in f if line.strip()]
	for i in range(0, len(lines), 3):
		layouts.append(Layout([literal_eval(l) for l in lines[i:i+3]]))
	return layouts

if __name__ == '__main__':
	signal.signal(signal.SIGINT, cleanup)
	try:
		if len(sys.argv) != 2:
			print(f"Usage: {sys.argv[0]} <result_path>")
			sys.exit(1)

		result_path = sys.argv[1]
		result_path = os.path.expanduser(result_path)
		result_path = os.path.abspath(result_path)
		os.makedirs(result_path, exist_ok=True)

		# Analyze
		file_path = os.path.join(result_path, ANALYZE_RESULT_FILENAME)
		if os.path.exists(file_path):
			load_analysis_result(result_path)
		else:
			analyze_target(result_path)

		# Optimize
		init_score_state()
		file_path = os.path.join(result_path, BEST_RESULT_FILENAME)
		if os.path.exists(file_path):
			best = load_best_result(result_path)
		else:
			best = []

		print(f'[Optimize]')
		unimproved = 0
		cur = [l.clone() for l in best]
		candy = [make_initial_layout()]
		while unimproved < 100:
			cur = optimize(candy)

			prev = [l.clone() for l in best]
			best.extend(cur)
			best = sort_unique_layouts(best)

			if prev and prev[-1] == best[-1]:
				unimproved += 1
				print(f'\t unimproved: {unimproved}')
			else:
				unimproved = 0
				save_best_result(best, result_path)
				print(f'\t improved')
				for i, l in enumerate(best, 1):
					print(f'[{i}]')
					print_layout(l)

			if unimproved > (BEST_SIZE/2):
				candy = [make_initial_layout()]
			else:
				candy = best

		print(f'\r\033[K...Done')

		best = sort_unique_layouts(best)
		save_best_result(best, result_path)
		for i, l in enumerate(best, 1):
			print(f'[{i}]')
			print_layout(l)

	except KeyboardInterrupt:
		cleanup(None, None)

	cleanup(None,None)
