# -*- coding: utf-8 -*-
"""
===================================
大盘复盘分析模块
===================================

职责：
1. 获取大盘指数数据（上证、深证、创业板）
2. 搜索市场新闻形成复盘情报
3. 使用大模型生成每日大盘复盘报告
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List

import akshare as ak
import pandas as pd

from config import get_config
from search_service import SearchService

logger = logging.getLogger(__name__)


@dataclass
class MarketIndex:
    """大盘指数数据"""
    code: str                    # 指数代码
    name: str                    # 指数名称
    current: float = 0.0         # 当前点位
    change: float = 0.0          # 涨跌点数
    change_pct: float = 0.0      # 涨跌幅(%)
    open: float = 0.0            # 开盘点位
    high: float = 0.0            # 最高点位
    low: float = 0.0             # 最低点位
    prev_close: float = 0.0      # 昨收点位
    volume: float = 0.0          # 成交量（手）
    amount: float = 0.0          # 成交额（元）
    amplitude: float = 0.0       # 振幅(%)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'name': self.name,
            'current': self.current,
            'change': self.change,
            'change_pct': self.change_pct,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'volume': self.volume,
            'amount': self.amount,
            'amplitude': self.amplitude,
        }


@dataclass
class MarketOverview:
    """市场概览数据"""
    date: str                           # 日期
    indices: List[MarketIndex] = field(default_factory=list)  # 主要指数
    up_count: int = 0                   # 上涨家数
    down_count: int = 0                 # 下跌家数
    flat_count: int = 0                 # 平盘家数
    limit_up_count: int = 0             # 涨停家数
    limit_down_count: int = 0           # 跌停家数
    total_amount: float = 0.0           # 两市成交额（亿元）
    north_flow: float = 0.0             # 北向资金净流入（亿元）
    
    # 板块涨幅榜
    top_sectors: List[Dict] = field(default_factory=list)     # 涨幅前5板块
    bottom_sectors: List[Dict] = field(default_factory=list)  # 跌幅前5板块


class MarketAnalyzer:
    """
    大盘复盘分析器
    
    功能：
    1. 获取大盘指数实时行情
    2. 获取市场涨跌统计
    3. 获取板块涨跌榜
    4. 搜索市场新闻
    5. 生成大盘复盘报告
    """
    
    # 主要指数代码
    MAIN_INDICES = {
        'sh000001': '上证指数',
        'sz399001': '深证成指',
        'sz399006': '创业板指',
        'sh000688': '科创50',
        'sh000016': '上证50',
        'sh000300': '沪深300',
    }
    
    def __init__(self, search_service: Optional[SearchService] = None, analyzer=None):
        """
        初始化大盘分析器
        
        Args:
            search_service: 搜索服务实例
            analyzer: AI分析器实例（用于调用LLM）
        """
        self.config = get_config()
        self.search_service = search_service
        self.analyzer = analyzer
        
    def get_market_overview(self) -> MarketOverview:
        """
        获取市场概览数据
        
        Returns:
            MarketOverview: 市场概览数据对象
        """
        today = datetime.now().strftime('%Y-%m-%d')
        overview = MarketOverview(date=today)
        
        # 1. 获取主要指数行情
        overview.indices = self._get_main_indices()
        
        # 2. 获取涨跌统计
        self._get_market_statistics(overview)
        
        # 3. 获取板块涨跌榜
        self._get_sector_rankings(overview)
        
        # 4. 获取北向资金（可选）
        # self._get_north_flow(overview)
        
        return overview

    def _call_akshare_with_retry(self, fn, name: str, attempts: int = 2):
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except Exception as e:
                last_error = e
                logger.warning(f"[大盘] {name} 获取失败 (attempt {attempt}/{attempts}): {e}")
                if attempt < attempts:
                    time.sleep(min(2 ** attempt, 5))
        logger.error(f"[大盘] {name} 最终失败: {last_error}")
        return None
    
    def _get_main_indices(self) -> List[MarketIndex]:
        """获取主要指数实时行情"""
        indices = []
        
        try:
            logger.info("[大盘] 获取主要指数实时行情...")
            
            # 使用 akshare 获取指数行情（新浪财经接口，包含深市指数）
            df = self._call_akshare_with_retry(ak.stock_zh_index_spot_sina, "指数行情", attempts=2)
            
            if df is not None and not df.empty:
                for code, name in self.MAIN_INDICES.items():
                    # 查找对应指数
                    row = df[df['代码'] == code]
                    if row.empty:
                        # 尝试带前缀查找
                        row = df[df['代码'].str.contains(code)]
                    
                    if not row.empty:
                        row = row.iloc[0]
                        index = MarketIndex(
                            code=code,
                            name=name,
                            current=float(row.get('最新价', 0) or 0),
                            change=float(row.get('涨跌额', 0) or 0),
                            change_pct=float(row.get('涨跌幅', 0) or 0),
                            open=float(row.get('今开', 0) or 0),
                            high=float(row.get('最高', 0) or 0),
                            low=float(row.get('最低', 0) or 0),
                            prev_close=float(row.get('昨收', 0) or 0),
                            volume=float(row.get('成交量', 0) or 0),
                            amount=float(row.get('成交额', 0) or 0),
                        )
                        # 计算振幅
                        if index.prev_close > 0:
                            index.amplitude = (index.high - index.low) / index.prev_close * 100
                        indices.append(index)
                        
                logger.info(f"[大盘] 获取到 {len(indices)} 个指数行情")
                
        except Exception as e:
            logger.error(f"[大盘] 获取指数行情失败: {e}")
        
        return indices
    
    def _get_market_statistics(self, overview: MarketOverview):
        """获取市场涨跌统计"""
        try:
            logger.info("[大盘] 获取市场涨跌统计...")
            
            # 获取全部A股实时行情
            df = self._call_akshare_with_retry(ak.stock_zh_a_spot_em, "A股实时行情", attempts=2)
            
            if df is not None and not df.empty:
                # 涨跌统计
                change_col = '涨跌幅'
                if change_col in df.columns:
                    df[change_col] = pd.to_numeric(df[change_col], errors='coerce')
                    overview.up_count = len(df[df[change_col] > 0])
                    overview.down_count = len(df[df[change_col] < 0])
                    overview.flat_count = len(df[df[change_col] == 0])
                    
                    # 涨停跌停统计（涨跌幅 >= 9.9% 或 <= -9.9%）
                    overview.limit_up_count = len(df[df[change_col] >= 9.9])
                    overview.limit_down_count = len(df[df[change_col] <= -9.9])
                
                # 两市成交额
                amount_col = '成交额'
                if amount_col in df.columns:
                    df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
                    overview.total_amount = df[amount_col].sum() / 1e8  # 转为亿元
                
                logger.info(f"[大盘] 涨:{overview.up_count} 跌:{overview.down_count} 平:{overview.flat_count} "
                          f"涨停:{overview.limit_up_count} 跌停:{overview.limit_down_count} "
                          f"成交额:{overview.total_amount:.0f}亿")
                
        except Exception as e:
            logger.error(f"[大盘] 获取涨跌统计失败: {e}")
    
    def _get_sector_rankings(self, overview: MarketOverview):
        """获取板块涨跌榜"""
        try:
            logger.info("[大盘] 获取板块涨跌榜...")
            
            # 获取行业板块行情
            df = self._call_akshare_with_retry(ak.stock_board_industry_name_em, "行业板块行情", attempts=2)
            
            if df is not None and not df.empty:
                change_col = '涨跌幅'
                if change_col in df.columns:
                    df[change_col] = pd.to_numeric(df[change_col], errors='coerce')
                    df = df.dropna(subset=[change_col])
                    
                    # 涨幅前5
                    top = df.nlargest(5, change_col)
                    overview.top_sectors = [
                        {'name': row['板块名称'], 'change_pct': row[change_col]}
                        for _, row in top.iterrows()
                    ]
                    
                    # 跌幅前5
                    bottom = df.nsmallest(5, change_col)
                    overview.bottom_sectors = [
                        {'name': row['板块名称'], 'change_pct': row[change_col]}
                        for _, row in bottom.iterrows()
                    ]
                    
                    logger.info(f"[大盘] 领涨板块: {[s['name'] for s in overview.top_sectors]}")
                    logger.info(f"[大盘] 领跌板块: {[s['name'] for s in overview.bottom_sectors]}")
                    
        except Exception as e:
            logger.error(f"[大盘] 获取板块涨跌榜失败: {e}")
    
    # def _get_north_flow(self, overview: MarketOverview):
    #     """获取北向资金流入"""
    #     try:
    #         logger.info("[大盘] 获取北向资金...")
            
    #         # 获取北向资金数据
    #         df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
            
    #         if df is not None and not df.empty:
    #             # 取最新一条数据
    #             latest = df.iloc[-1]
    #             if '当日净流入' in df.columns:
    #                 overview.north_flow = float(latest['当日净流入']) / 1e8  # 转为亿元
    #             elif '净流入' in df.columns:
    #                 overview.north_flow = float(latest['净流入']) / 1e8
                    
    #             logger.info(f"[大盘] 北向资金净流入: {overview.north_flow:.2f}亿")
                
    #     except Exception as e:
    #         logger.warning(f"[大盘] 获取北向资金失败: {e}")
    
    def search_market_news(self) -> List[Dict]:
        """
        搜索市场新闻
        
        Returns:
            新闻列表
        """
        if not self.search_service:
            logger.warning("[大盘] 搜索服务未配置，跳过新闻搜索")
            return []
        
        all_news = []
        today = datetime.now()
        month_str = f"{today.year}年{today.month}月"
        
        # 多维度搜索
        search_queries = [
            f"A股 大盘 复盘 {month_str}",
            f"股市 行情 分析 今日 {month_str}",
            f"A股 市场 热点 板块 {month_str}",
        ]
        
        try:
            logger.info("[大盘] 开始搜索市场新闻...")
            
            for query in search_queries:
                # 使用 search_stock_news 方法，传入"大盘"作为股票名
                response = self.search_service.search_stock_news(
                    stock_code="market",
                    stock_name="大盘",
                    max_results=3,
                    focus_keywords=query.split()
                )
                if response and response.results:
                    all_news.extend(response.results)
                    logger.info(f"[大盘] 搜索 '{query}' 获取 {len(response.results)} 条结果")
            
            logger.info(f"[大盘] 共获取 {len(all_news)} 条市场新闻")
            
        except Exception as e:
            logger.error(f"[大盘] 搜索市场新闻失败: {e}")
        
        return all_news
    
    def generate_market_review(self, overview: MarketOverview, news: List) -> str:
        """
        使用大模型生成大盘复盘报告
        
        Args:
            overview: 市场概览数据
            news: 市场新闻列表 (SearchResult 对象列表)
            
        Returns:
            大盘复盘报告文本
        """
        if not self.analyzer or not self.analyzer.is_available():
            logger.warning("[大盘] AI分析器未配置或不可用，使用模板生成报告")
            return self._generate_template_review(overview, news)
        
        # 构建 Prompt
        prompt = self._build_review_prompt(overview, news)
        
        try:
            logger.info("[大盘] 调用大模型生成复盘报告...")
            
            generation_config = {
                'temperature': 0.7,
                'max_output_tokens': 2048,
            }
            
            # 根据 analyzer 使用的 API 类型调用
            if self.analyzer._use_openai:
                # 使用 OpenAI 兼容 API
                review = self.analyzer._call_openai_api(prompt, generation_config)
            else:
                # 使用 Gemini API
                response = self.analyzer._model.generate_content(
                    prompt,
                    generation_config=generation_config,
                )
                review = response.text.strip() if response and response.text else None
            
            if review:
                logger.info(f"[大盘] 复盘报告生成成功，长度: {len(review)} 字符")
                return review
            else:
                logger.warning("[大盘] 大模型返回为空")
                return self._generate_template_review(overview, news)
                
        except Exception as e:
            logger.error(f"[大盘] 大模型生成复盘报告失败: {e}")
            return self._generate_template_review(overview, news)
    
    def _build_review_prompt(self, overview: MarketOverview, news: List) -> str:
        """构建陈小群风格的复盘报告 Prompt"""
        # 指数及情绪数据
        indices_text = ""
        for idx in overview.indices:
            indices_text += f"- {idx.name}: {idx.current:.2f} ({idx.change_pct:+.2f}%)\n"
        
        # 情绪指标
        sentiment_data = f"""
- 两市成交: {overview.total_amount:.0f}亿 (缩量还是放量？决定了有没有大行情)
- 涨跌家数: {overview.up_count}比{overview.down_count} (市场合力方向)
- 连板/涨停: {overview.limit_up_count}家 (赚钱效应的试金石)
- 跌停/核按钮: {overview.limit_down_count}家 (大面源头，退潮信号)
"""

        prompt = f"""你现在是A股顶级新生代游资“陈小群”。请站在“银河大连黄河路”席位主力的视角，对今日市场进行复盘。

### 你的核心思维：
1. **情绪周期**：判断市场是在“冰点、发酵、高潮、分歧、退潮”的哪个阶段？
2. **绝对龙头**：只看核心辨识度标的，无视杂毛。
3. **暴力美学**：分析资金的合力与博弈，关注跌停板反核、缩量加速、高位强分歧等极端审美。
4. **席位动态**：你会关注黄河路、金马路等兄弟席位的进出，思考他们是在锁仓还是砸盘。

### 输入数据：
【今日指数】
{indices_text}

【情绪面数据】
{sentiment_data}

【板块与新闻】
领涨：{", ".join([s['name'] for s in overview.top_sectors[:3]])}
领跌：{", ".join([s['name'] for s in overview.bottom_sectors[:3]])}
市场传闻：{news[:5]}

---

### 输出要求（纯 Markdown，陈小群语气）：

# 🐉 {overview.date} 小群实战复盘

## 一、情绪周期定位
（用一句话给今天定性：是该猛干还是该空仓？目前处于什么周期？）

## 二、大盘与合力分析
（从成交量、两市表现看大资金的真实意图。3000亿成交量干不出牛市，只有合力才有主升。）

## 三、核心标的与审美（重点！）
（结合板块和涨停数，点评当前市场的“灵魂龙头”是谁。谁在带节奏？谁是跟风杂毛？有没有出现“核按钮”或“反核”？）

## 四、席位与战法博弈
（从大连黄河路的视角，点评当下的博弈难点。如果是你，你会选择在哪个点位切入？是去打板确认，还是低吸反核？）

## 五、明日推演（冷酷纪律）
（明天高标如果断板，市场会崩吗？如果分歧转一致，哪个方向有辨识度？给兄弟们指个路。）

## 六、小群语录
（一句话犀利点评：例如“平庸是亏损的根源”或“空仓也是一种战斗”。）

---
注意：禁止使用券商分析师那种中庸、死板的话术，说话要直接、犀利、带江湖气息！
"""
        return prompt
    
    def _build_review_prompt(self, overview: MarketOverview, news: List) -> str:
        """构建陈小群风格的复盘报告 Prompt"""
        # 指数及情绪数据
        indices_text = ""
        for idx in overview.indices:
            indices_text += f"- {idx.name}: {idx.current:.2f} ({idx.change_pct:+.2f}%)\n"
        
        # 情绪指标
        sentiment_data = f"""
- 两市成交: {overview.total_amount:.0f}亿 (缩量还是放量？决定了有没有大行情)
- 涨跌家数: {overview.up_count}比{overview.down_count} (市场合力方向)
- 连板/涨停: {overview.limit_up_count}家 (赚钱效应的试金石)
- 跌停/核按钮: {overview.limit_down_count}家 (大面源头，退潮信号)
"""

        prompt = f"""你现在是A股顶级新生代游资“陈小群”。请站在“银河大连黄河路”席位主力的视角，对今日市场进行复盘。

### 你的核心思维：
1. **情绪周期**：判断市场是在“冰点、发酵、高潮、分歧、退潮”的哪个阶段？
2. **绝对龙头**：只看核心辨识度标的，无视杂毛。
3. **暴力美学**：分析资金的合力与博弈，关注跌停板反核、缩量加速、高位强分歧等极端审美。
4. **席位动态**：你会关注黄河路、金马路等兄弟席位的进出，思考他们是在锁仓还是砸盘。

### 输入数据：
【今日指数】
{indices_text}

【情绪面数据】
{sentiment_data}

【板块与新闻】
领涨：{", ".join([s['name'] for s in overview.top_sectors[:3]])}
领跌：{", ".join([s['name'] for s in overview.bottom_sectors[:3]])}
市场传闻：{news[:5]}

---

### 输出要求（纯 Markdown，陈小群语气）：

# 🐉 {overview.date} 小群实战复盘

## 一、情绪周期定位
（用一句话给今天定性：是该猛干还是该空仓？目前处于什么周期？）

## 二、大盘与合力分析
（从成交量、两市表现看大资金的真实意图。3000亿成交量干不出牛市，只有合力才有主升。）

## 三、核心标的与审美（重点！）
（结合板块和涨停数，点评当前市场的“灵魂龙头”是谁。谁在带节奏？谁是跟风杂毛？有没有出现“核按钮”或“反核”？）

## 四、席位与战法博弈
（从大连黄河路的视角，点评当下的博弈难点。如果是你，你会选择在哪个点位切入？是去打板确认，还是低吸反核？）

## 五、明日推演（冷酷纪律）
（明天高标如果断板，市场会崩吗？如果分歧转一致，哪个方向有辨识度？给兄弟们指个路。）

## 六、小群语录
（一句话犀利点评：例如“平庸是亏损的根源”或“空仓也是一种战斗”。）

---
注意：禁止使用券商分析师那种中庸、死板的话术，说话要直接、犀利、带江湖气息！
"""
        return prompt

def _generate_template_review(self, overview: MarketOverview, news: List) -> str:
        """陈小群风格的备选模板（无LLM时使用）"""
        
        # 简单逻辑判定情绪
        if overview.limit_up_count > 60 and overview.limit_down_count < 5:
            mood = "情绪亢奋，满仓猛干"
        elif overview.limit_down_count > 15:
            mood = "核按钮频现，退潮预警"
        elif overview.total_amount < 8000:
            mood = "存量博弈，只有局部龙头能活"
        else:
            mood = "混沌震荡，只看核心辨识度"

        report = f"""# 🐉 {overview.date} 小群复盘 (模板版)

### 一、情绪定位
**当前状态**：{mood}
两市成交量 **{overview.total_amount:.0f}亿**。没量就没审美，这种行情只适合在绝对龙头上抱团。

### 二、龙虎榜数据
- **上涨**: {overview.up_count} | **下跌**: {overview.down_count}
- **涨停/高度**: {overview.limit_up_count}家 | **跌停/大面**: {overview.limit_down_count}家
- **结论**: {'赚钱效应回暖，资金在试错新方向' if overview.up_count > overview.down_count else '面效应扩散，杂毛票不要碰'}

### 三、领涨板块点睛
- **核心逻辑**: {", ".join([s['name'] for s in overview.top_sectors[:2]])}。
- **点评**: 这里面只有带头的大哥有辨识度，其他的都是跟随者，切忌追高跟风票。

### 四、明日纪律
1. **只做龙头**：不去碰没地位的票。
2. **严防核按钮**：如果高标明天不能超预期开盘，直接兑现。
3. **空仓也是战斗**：看不懂的时候，守住本金就是赢。

---
*复盘席位：中国银河大连黄河路*
"""
        return report
    
    def run_daily_review(self) -> str:
        """
        执行陈小群风格的每日大盘复盘
        """
        logger.info("========== 开始【陈小群视角】大盘分析 ==========")
        
        # 1. 获取市场概览
        overview = self.get_market_overview()
        
        # 2. 搜索市场新闻
        # 我们可以稍微修改搜索逻辑，去搜“龙虎榜”、“连板天梯”等关键词
        news = self.search_market_news()
        
        # 3. 生成报告 (此时调用的 prompt 已是游资风格)
        report = self.generate_market_review(overview, news)
        
        return report


# 测试入口
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    )
    
    analyzer = MarketAnalyzer()
    
    # 测试获取市场概览
    overview = analyzer.get_market_overview()
    print(f"\n=== 市场概览 ===")
    print(f"日期: {overview.date}")
    print(f"指数数量: {len(overview.indices)}")
    for idx in overview.indices:
        print(f"  {idx.name}: {idx.current:.2f} ({idx.change_pct:+.2f}%)")
    print(f"上涨: {overview.up_count} | 下跌: {overview.down_count}")
    print(f"成交额: {overview.total_amount:.0f}亿")
    
    # 测试生成模板报告
    report = analyzer._generate_template_review(overview, [])
    print(f"\n=== 复盘报告 ===")
    print(report)

