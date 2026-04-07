#!/usr/bin/env python3
"""
高难度推理能力评测 - 奥赛级别问题
包含：数学奥林匹克、逻辑推理、组合数学、数论等
答案确定，难度极高
"""

import requests
import json
import time

API_BASE = "http://127.0.0.1:1234/v1"
MODEL_ID = "gemma-4-31b-it"

def chat_completion(prompt: str, max_tokens: int = 4096, temperature: float = 0.1) -> str:
    """调用 chat completion API，低温度保证一致性"""
    resp = requests.post(
        f"{API_BASE}/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=600
    )
    return resp.json()["choices"][0]["message"]["content"]

# 定义10个高难度问题
HARD_QUESTIONS = [
    {
        "id": 1,
        "category": "数论",
        "difficulty": "IMO",
        "question": """证明：对于任意正整数 n，n³ - n 能被 6 整除。
请给出严格的数学证明过程。""",
        "answer_key": "n³ - n = n(n-1)(n+1)，三个连续整数必有一个被3整除，必有一个被2整除，所以被6整除",
        "key_points": ["连续整数", "n(n-1)(n+1)", "被2整除", "被3整除", "分解因式"]
    },
    {
        "id": 2,
        "category": "组合数学",
        "difficulty": "IMO",
        "question": """将数字 1, 2, 3, 4, 5, 6 填入一个 2×3 的表格中，每个数字恰好使用一次。
要求：每行从左到右递增，每列从上到下递增。
问：有多少种不同的填法？请给出详细的计数过程。""",
        "answer_key": "5种。这是标准Young表问题，2×3矩形的标准Young表数量为C(6,3)/(4×3×2×1) = 5",
        "key_points": ["5种", "Young表", "C(6,3)", "钩子公式", "递增"]
    },
    {
        "id": 3,
        "category": "几何",
        "difficulty": "IMO",
        "question": """在三角形 ABC 中，∠A = 60°，BC = 7，AC = 8。
求 AB 的长度。请给出完整的计算过程。""",
        "answer_key": "AB = 5。由余弦定理：BC² = AB² + AC² - 2·AB·AC·cos(60°)，49 = AB² + 64 - 8·AB，解得AB = 5或AB = 3，但AB = 3时无法构成三角形，所以AB = 5",
        "key_points": ["余弦定理", "cos(60°)=1/2", "AB=5", "方程", "检验"]
    },
    {
        "id": 4,
        "category": "数列与递推",
        "difficulty": "IMO",
        "question": """定义数列 {aₙ}：a₁ = 1，a₂ = 2，对于 n ≥ 3，aₙ = aₙ₋₁ + aₙ₋₂。
求 a₁ + a₃ + a₅ + a₇ + a₉ 的值。
请给出推理过程。""",
        "answer_key": "数列为斐波那契数列：1, 2, 3, 5, 8, 13, 21, 34, 55... 奇数项为a₁=1, a₃=3, a₅=8, a₇=21, a₉=55，和为1+3+8+21+55=88",
        "key_points": ["斐波那契", "88", "1,2,3,5,8,13,21,34,55", "奇数项", "求和"]
    },
    {
        "id": 5,
        "category": "概率论",
        "difficulty": "竞赛",
        "question": """一个袋子中有 10 个球，编号 1 到 10。随机取出 3 个球。
求这 3 个球编号之和为奇数的概率。
请给出详细的计算过程。""",
        "answer_key": "3个数和为奇数，需要(奇+奇+奇)或(奇+偶+偶)。1-10中奇数5个，偶数5个。方法数=C(5,3)+C(5,1)×C(5,2)=10+50=60。总方法数=C(10,3)=120。概率=60/120=1/2",
        "key_points": ["1/2", "奇+奇+奇", "奇+偶+偶", "C(5,3)", "C(10,3)", "60/120"]
    },
    {
        "id": 6,
        "category": "逻辑推理",
        "difficulty": "竞赛",
        "question": """有 100 个囚犯和 100 个盒子。每个盒子里放着一个囚犯的名字（每个名字恰好出现一次）。
囚犯们依次进入房间，每人可以打开 50 个盒子。如果所有囚犯都找到了自己的名字，则全部释放。
囚犯们在开始前可以商量策略，但进入房间后不能交流。
问：最优策略下，所有囚犯都被释放的概率大约是多少？请解释这个策略。""",
        "answer_key": "约31%。策略：每个囚犯先打开写有自己名字的盒子，然后打开盒子中名字对应的盒子，依此形成循环。最优情况下成功率约1-ln2≈31%。如果随机打开，成功率仅为(1/2)^100≈0",
        "key_points": ["循环策略", "31%", "1-ln2", "约0.31", "置换循环"]
    },
    {
        "id": 7,
        "category": "代数",
        "difficulty": "IMO",
        "question": """求方程 x² + y² = 2025 的所有正整数解 (x, y)。
请给出完整的求解过程。""",
        "answer_key": "2025 = 45²。设x² + y² = 45²，需要找半径45的圆周上的整数点。分解2025 = 3⁴×5²。利用勾股数性质，检验所有x从1到45。实际上45² = 2025，检查哪些平方和等于2025：(27,36)因为27²+36²=729+1296=2025，(36,27)同理。还有(9,?)，9²=81，2025-81=1944不是平方数。答案：(27,36), (36,27)",
        "key_points": ["27,36", "36,27", "2025=45²", "勾股数", "平方和"]
    },
    {
        "id": 8,
        "category": "组合博弈",
        "difficulty": "竞赛",
        "question": """有一堆 n 个石子。两个玩家轮流取走石子，每次可以取走 1、2 或 3 个石子。
取走最后一个石子的玩家获胜。
问：当 n = 20 时，先手必胜还是后手必胜？请给出完整的博弈分析。""",
        "answer_key": "后手必胜。分析：若剩余石子数是4的倍数，则当前玩家必败（无论取1、2、3，对手都能凑成4）。20 = 4×5，是4的倍数，所以先手必败，后手必胜。策略：后手始终保持剩余石子数为4的倍数",
        "key_points": ["后手必胜", "4的倍数", "模4", "必败态", "策略"]
    },
    {
        "id": 9,
        "category": "不等式",
        "difficulty": "IMO",
        "question": """证明：对于任意正实数 a, b, c，有
a/(b+c) + b/(c+a) + c/(a+b) ≥ 3/2

请给出完整的证明过程。""",
        "answer_key": "这是Nesbitt不等式。证明方法：设S = a/(b+c) + b/(c+a) + c/(a+b)。由对称性可设a≤b≤c。方法一：通分后使用柯西不等式。方法二：(a+b+c)(1/(b+c)+1/(c+a)+1/(a+b))≥9/2，然后展开。方法三：a/(b+c)+1 = (a+b+c)/(b+c)，所以S+3 = (a+b+c)(1/(a+b)+1/(b+c)+1/(c+a)) ≥ (a+b+c)×9/(2(a+b+c)) = 9/2，因此S≥3/2",
        "key_points": ["Nesbitt", "3/2", "柯西不等式", "对称性", "通分"]
    },
    {
        "id": 10,
        "category": "图论",
        "difficulty": "竞赛",
        "question": """一个完全图 K_n 有 n 个顶点，每两个顶点之间都有一条边。
现在要将所有边染成红色或蓝色。
证明：当 n ≥ 6 时，必然存在一个同色的三角形（三条边颜色相同）。
请给出完整的证明。""",
        "answer_key": "这是拉姆齐定理的特殊情况R(3,3)=6。证明：取任意顶点v，v有5条边连向其他顶点。由鸽巢原理，至少有3条边同色（比如红色），设这3条边连接顶点a,b,c。若a,b,c之间有任何红边（如ab是红边），则v,a,b形成红色三角形。若a,b,c之间全为蓝边，则a,b,c形成蓝色三角形。因此必存在同色三角形",
        "key_points": ["拉姆齐", "R(3,3)=6", "鸽巢原理", "同色三角形", "必存在"]
    }
]

def evaluate_answer(response: str, answer_key: str, key_points: list) -> dict:
    """评估回答"""
    response_lower = response.lower()

    # 检查关键点
    found_points = []
    for point in key_points:
        if point.lower() in response_lower:
            found_points.append(point)

    score = len(found_points) / len(key_points) if key_points else 0

    return {
        "found_points": found_points,
        "score": score,
        "answer_key": answer_key
    }

def run_benchmark():
    print("=" * 80)
    print("高难度推理能力评测 - 奥赛级别问题")
    print(f"模型: {MODEL_ID}")
    print("=" * 80)

    results = []

    for q in HARD_QUESTIONS:
        print(f"\n{'='*80}")
        print(f"问题 {q['id']}: [{q['category']}] 难度: {q['difficulty']}")
        print("=" * 80)
        print(f"\n【题目】\n{q['question']}")

        print(f"\n思考中...")
        start_time = time.time()

        try:
            response = chat_completion(q['question'])
            elapsed = time.time() - start_time

            print(f"\n【模型回答】\n{response}")

            # 评估
            eval_result = evaluate_answer(response, q['answer_key'], q['key_points'])

            print(f"\n【参考答案】\n{q['answer_key']}")
            print(f"\n【评估】")
            print(f"  关键点命中: {eval_result['found_points']}")
            print(f"  得分: {eval_result['score']:.0%}")
            print(f"  耗时: {elapsed:.1f}s")

            results.append({
                "id": q['id'],
                "category": q['category'],
                "score": eval_result['score'],
                "time": elapsed,
                "found_points": eval_result['found_points']
            })

        except Exception as e:
            print(f"\n【错误】{str(e)}")
            results.append({
                "id": q['id'],
                "category": q['category'],
                "score": 0,
                "time": 0,
                "found_points": [],
                "error": str(e)
            })

    # 汇总
    print("\n" + "=" * 80)
    print("评测汇总")
    print("=" * 80)

    print(f"\n{'题号':<6} {'类别':<12} {'得分':<10} {'耗时':<10} {'命中关键点'}")
    print("-" * 80)

    total_score = 0
    total_time = 0

    for r in results:
        pts = ', '.join(r['found_points'][:3]) if r['found_points'] else '-'
        print(f"{r['id']:<6} {r['category']:<12} {r['score']:.0%}        {r['time']:.1f}s       {pts}")
        total_score += r['score']
        total_time += r['time']

    print("-" * 80)
    avg_score = total_score / len(results)
    print(f"平均得分: {avg_score:.0%}")
    print(f"总耗时: {total_time:.1f}s")

    # 分类统计
    print("\n【分类统计】")
    categories = {}
    for r in results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r['score'])

    for cat, scores in sorted(categories.items()):
        avg = sum(scores) / len(scores)
        print(f"  {cat}: {avg:.0%}")

if __name__ == "__main__":
    run_benchmark()
