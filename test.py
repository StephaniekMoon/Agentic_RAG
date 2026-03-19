# test_web_search.py
import os
from dotenv import load_dotenv
load_dotenv()

from src.agentic_rag.tools.custom_tool import FireCrawlWebSearchTool

def test_web_search():
    """测试网页搜索工具"""
    print("🔧 初始化网页搜索工具...")
    try:
        tool = FireCrawlWebSearchTool()
        print("✅ 工具初始化成功！")
        
        # 测试搜索
        query = "DSPy framework introduction"
        print(f"\n🔍 测试搜索: '{query}'")
        
        result = tool._run(query, limit=2)
        
        print("\n📄 搜索结果:")
        print("-" * 50)
        print(result[:1000] + "..." if len(result) > 1000 else result)
        print("-" * 50)
        print("✅ 搜索完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    test_web_search()