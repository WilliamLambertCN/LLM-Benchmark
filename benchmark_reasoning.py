#!/usr/bin/env python3
"""
模型推理能力评测脚本
测试逻辑推理、数学推理、代码推理等能力
"""

import requests
import json
import time

API_BASE = "http://127.0.0.1:1234/v1"
MODEL_ID = "gemma-4-31b-it"

def chat_completion(prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
    """调用 chat completion API"""
    resp = requests.post(
        f"{API_BASE}/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=300
    )
    return resp.json()["choices"][0]["message"]["content"]

def evaluate_response(response: str, keywords: list, exact_match: str = None) -> dict:
    """简单评估回答"""
    response_lower = response.lower()
    found_keywords = [k for k in keywords if k.lower() in response_lower]

    if exact_match:
        exact_found = exact_match.lower() in response_lower
    else:
        exact_found = None

    return {
        "found_keywords": found_keywords,
        "keyword_score": len(found_keywords) / len(keywords) if keywords else 0,
        "exact_match": exact_found
    }

def run_evaluation():
    print("=" * 70)
    print(f"模型推理能力评测 - {MODEL_ID}")
    print("=" * 70)

    results = []

    # ========== 1. 逻辑推理 ==========
    print("\n" + "=" * 70)
    print("1. 逻辑推理测试")
    print("=" * 70)

    logic_tests = [
        {
            "name": "Sally 有多少个兄弟",
            "prompt": """Sally 有 3 个兄弟。每个兄弟都有 2 个姐妹。
请问 Sally 有多少个姐妹？请一步步推理，然后给出答案。""",
            "expected_answer": "1",
            "keywords": ["1个", "一个姐妹", "sally自己", "就是sally"]
        },
        {
            "name": "过河问题",
            "prompt": """一个农夫需要把一只狼、一只羊和一颗白菜运过河。
他的小船只能容纳他自己和一样东西。
如果他留下狼和羊在一起，狼会吃羊。
如果他留下羊和白菜在一起，羊会吃白菜。
请问农夫如何才能把这三样东西都安全运到对岸？请详细描述步骤。""",
            "expected_answer": None,
            "keywords": ["带羊", "带狼", "带白菜", "回程", "对岸", "第一步", "第二步"]
        },
        {
            "name": "说谎者悖论",
            "prompt": """一个岛上住着两类人：骑士总是说真话，无赖总是说谎。
你遇到了两个人 A 和 B。
A 说："我们两个都是无赖。"
请问 A 和 B 分别是什么身份？请给出推理过程。""",
            "expected_answer": "A是骑士，B是无赖" if False else "A是无赖",
            "keywords": ["无赖", "骑士", "矛盾", "假设", "不可能"]
        }
    ]

    for test in logic_tests:
        print(f"\n【{test['name']}】")
        print(f"问题: {test['prompt'][:100]}...")
        start = time.time()
        response = chat_completion(test['prompt'])
        elapsed = time.time() - start
        print(f"\n回答:\n{response[:500]}{'...' if len(response) > 500 else ''}")
        print(f"\n⏱️ 耗时: {elapsed:.1f}s")

        eval_result = evaluate_response(response, test['keywords'], test.get('expected_answer'))
        print(f"📊 关键词命中: {eval_result['found_keywords']} (得分: {eval_result['keyword_score']:.0%})")

        results.append({
            "category": "逻辑推理",
            "name": test['name'],
            "score": eval_result['keyword_score'],
            "time": elapsed
        })

    # ========== 2. 数学推理 ==========
    print("\n" + "=" * 70)
    print("2. 数学推理测试")
    print("=" * 70)

    math_tests = [
        {
            "name": "鸡兔同笼",
            "prompt": """鸡兔同笼，共有 35 个头，94 只脚。
请问鸡和兔各有多少只？请列出方程并求解。""",
            "keywords": ["鸡", "兔", "23", "12", "方程", "解得"]
        },
        {
            "name": "概率问题",
            "prompt": """一个袋子里有 3 个红球和 2 个蓝球。
如果不放回地连续取出 2 个球，两个都是红球的概率是多少？
请给出详细的计算过程。""",
            "keywords": ["3/10", "0.3", "30%", "3/5", "2/4", "第一步", "第二步"]
        },
        {
            "name": "数列推理",
            "prompt": """找出下列数列的规律，并预测下一个数字：
2, 6, 12, 20, 30, ?
请解释你的推理过程。""",
            "keywords": ["42", "差", "+4", "+6", "+8", "+10", "n(n+1)"]
        }
    ]

    for test in math_tests:
        print(f"\n【{test['name']}】")
        print(f"问题: {test['prompt'][:100]}...")
        start = time.time()
        response = chat_completion(test['prompt'])
        elapsed = time.time() - start
        print(f"\n回答:\n{response[:500]}{'...' if len(response) > 500 else ''}")
        print(f"\n⏱️ 耗时: {elapsed:.1f}s")

        eval_result = evaluate_response(response, test['keywords'])
        print(f"📊 关键词命中: {eval_result['found_keywords']} (得分: {eval_result['keyword_score']:.0%})")

        results.append({
            "category": "数学推理",
            "name": test['name'],
            "score": eval_result['keyword_score'],
            "time": elapsed
        })

    # ========== 3. 常识推理 ==========
    print("\n" + "=" * 70)
    print("3. 常识推理测试")
    print("=" * 70)

    common_tests = [
        {
            "name": "时间推理",
            "prompt": """如果现在是北京时间下午 3 点，那么：
1. 纽约时间是几点？（纽约比北京晚 13 小时）
2. 伦敦时间是几点？（伦敦比北京晚 8 小时）
3. 东京时间是几点？（东京比北京早 1 小时）
请给出计算过程。""",
            "keywords": ["纽约", "凌晨", "2点", "伦敦", "上午", "7点", "东京", "下午", "4点"]
        },
        {
            "name": "因果推理",
            "prompt": """小明每天早上 7 点起床，8 点到学校。
今天他 7:30 才起床。
请推理：他今天会发生什么？可能有哪些后果？""",
            "keywords": ["迟到", "赶不上", "公交", "错过", "后果", "可能"]
        }
    ]

    for test in common_tests:
        print(f"\n【{test['name']}】")
        print(f"问题: {test['prompt'][:100]}...")
        start = time.time()
        response = chat_completion(test['prompt'])
        elapsed = time.time() - start
        print(f"\n回答:\n{response[:500]}{'...' if len(response) > 500 else ''}")
        print(f"\n⏱️ 耗时: {elapsed:.1f}s")

        eval_result = evaluate_response(response, test['keywords'])
        print(f"📊 关键词命中: {eval_result['found_keywords']} (得分: {eval_result['keyword_score']:.0%})")

        results.append({
            "category": "常识推理",
            "name": test['name'],
            "score": eval_result['keyword_score'],
            "time": elapsed
        })

    # ========== 4. 代码推理 ==========
    print("\n" + "=" * 70)
    print("4. 代码推理测试")
    print("=" * 70)

    code_tests = [
        {
            "name": "代码纠错",
            "prompt": """以下 Python 代码有什么问题？请指出错误并修正：

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(50))

这段代码运行会很慢，请解释原因并给出优化版本。""",
            "keywords": ["重复计算", "递归", "缓存", "memo", "动态规划", "迭代", "O(2^n)"]
        },
        {
            "name": "代码输出预测",
            "prompt": """请预测以下 Python 代码的输出，并解释原因：

x = [1, 2, 3]
y = x
y.append(4)
print(x)
print(x is y)

z = x.copy()
z.append(5)
print(x)
print(z)
""",
            "keywords": ["[1, 2, 3, 4]", "True", "[1, 2, 3, 4]", "引用", "复制", "浅拷贝"]
        }
    ]

    for test in code_tests:
        print(f"\n【{test['name']}】")
        print(f"问题: {test['prompt'][:100]}...")
        start = time.time()
        response = chat_completion(test['prompt'])
        elapsed = time.time() - start
        print(f"\n回答:\n{response[:600]}{'...' if len(response) > 600 else ''}")
        print(f"\n⏱️ 耗时: {elapsed:.1f}s")

        eval_result = evaluate_response(response, test['keywords'])
        print(f"📊 关键词命中: {eval_result['found_keywords']} (得分: {eval_result['keyword_score']:.0%})")

        results.append({
            "category": "代码推理",
            "name": test['name'],
            "score": eval_result['keyword_score'],
            "time": elapsed
        })

    # ========== 5. 创意推理 ==========
    print("\n" + "=" * 70)
    print("5. 创意推理测试")
    print("=" * 70)

    creative_tests = [
        {
            "name": "脑筋急转弯",
            "prompt": """什么东西越洗越脏？请给出答案并解释。
然后再举 3 个类似的脑筋急转弯例子。""",
            "keywords": ["水", "脏", "洗澡", "洗澡水", "答案"]
        },
        {
            "name": "发散思维",
            "prompt": """请列出砖头的 10 种非常规用途（不能用来盖房子）。
要求创意性强，每种用途用一句话说明。""",
            "keywords": ["门挡", "书架", "健身", "装饰", "垫", "武器", "尺子", "镇纸"]
        }
    ]

    for test in creative_tests:
        print(f"\n【{test['name']}】")
        print(f"问题: {test['prompt'][:100]}...")
        start = time.time()
        response = chat_completion(test['prompt'])
        elapsed = time.time() - start
        print(f"\n回答:\n{response[:500]}{'...' if len(response) > 500 else ''}")
        print(f"\n⏱️ 耗时: {elapsed:.1f}s")

        eval_result = evaluate_response(response, test['keywords'])
        print(f"📊 关键词命中: {eval_result['found_keywords']} (得分: {eval_result['keyword_score']:.0%})")

        results.append({
            "category": "创意推理",
            "name": test['name'],
            "score": eval_result['keyword_score'],
            "time": elapsed
        })

    # ========== 汇总 ==========
    print("\n" + "=" * 70)
    print("评测汇总")
    print("=" * 70)

    categories = {}
    for r in results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r['score'])

    print(f"\n{'类别':<15} {'平均得分':<10} {'测试数':<8}")
    print("-" * 40)
    total_score = 0
    total_count = 0
    for cat, scores in categories.items():
        avg = sum(scores) / len(scores)
        print(f"{cat:<15} {avg:.0%}       {len(scores)}")
        total_score += sum(scores)
        total_count += len(scores)

    print("-" * 40)
    print(f"{'总体':<15} {total_score/total_count:.0%}       {total_count}")

    total_time = sum(r['time'] for r in results)
    print(f"\n总耗时: {total_time:.1f}s")

if __name__ == "__main__":
    run_evaluation()
