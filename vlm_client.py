"""
VLM 客户端包装器 - 支持原生 Dashscope 和 OpenAI 兼容模式
"""

from typing import List, Dict, Any
import time

try:
    from dashscope import MultiModalConversation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False

from openai import OpenAI


class _CompletionsAdapter:
    def __init__(self, owner: "VLMClient"):
        self.owner = owner

    def create(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        return self.owner.chat_completions_create(messages=messages, **kwargs)


class _ChatAdapter:
    def __init__(self, owner: "VLMClient"):
        self.completions = _CompletionsAdapter(owner)


# 简单的响应对象，用于统一 Dashscope 和 OpenAI 的返回格式
class SimpleMessage:
    """简单的消息对象"""
    def __init__(self, content):
        self.content = content


class SimpleChoice:
    """简单的选项对象"""
    def __init__(self, content):
        self.message = SimpleMessage(content)


class SimpleResponse:
    """简单的响应对象，模仿 OpenAI 响应"""
    def __init__(self, content):
        self.choices = [SimpleChoice(content)]


class VLMClient:
    """统一的 VLM 客户端接口，自动选择最佳实现"""
    
    def __init__(self, api_key: str, base_url: str, model_name: str):
        """
        初始化 VLM 客户端
        
        Args:
            api_key: API 密钥
            base_url: API 基础 URL (用于检测使用哪个提供商)
            model_name: 模型名称
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.chat = _ChatAdapter(self)
        
        # 检测提供商并选择实现
        if "dashscope" in base_url.lower():
            if DASHSCOPE_AVAILABLE:
                print(f"[VLMClient] Using native Dashscope SDK")
                self.client_type = "dashscope"
                # 设置 Dashscope API Key
                import dashscope
                dashscope.api_key = api_key
            else:
                print(f"[VLMClient] Dashscope SDK unavailable, fallback to OpenAI mode")
                self.client_type = "openai"
                self.openai_client = OpenAI(
                    api_key=api_key,
                    base_url=base_url
                )
        else:
            self.client_type = "openai"
            self.openai_client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
    
    def chat_completions_create(self, 
                               messages: List[Dict[str, Any]],
                               **kwargs) -> Any:
        """
        创建聊天完成请求，统一接口
        """
        model_name = kwargs.pop("model", None) or self.model_name
        if self.client_type == "dashscope":
            return self._create_dashscope(messages, model_name=model_name, **kwargs)
        else:
            return self._create_openai(messages, model_name=model_name, **kwargs)
    
    def _create_dashscope(self, messages: List[Dict[str, Any]], model_name: str, **kwargs) -> Any:
        """使用原生 Dashscope SDK"""
        try:
            from dashscope import MultiModalConversation
            
            # 合并 system 和 user 消息（Dashscope 可能不支持 system role）
            system_content = ""
            user_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg.get("content", "")
                elif msg["role"] == "user":
                    user_messages.append(msg)
            
            # 如果有 system 消息，将其合并到第一条 user 消息
            if system_content and user_messages:
                content = user_messages[0].get("content", [])
                if isinstance(content, list):
                    content.insert(0, {"type": "text", "text": system_content})
                    user_messages[0]["content"] = content
            
            # 转换消息格式为 Dashscope 格式
            dashscope_messages = []
            for msg in user_messages:
                content = msg.get("content", [])
                
                # 处理内容
                dashscope_content = []
                if isinstance(content, str):
                    dashscope_content.append({"text": content})
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, str):
                            dashscope_content.append({"text": item})
                        elif isinstance(item, dict):
                            if item.get("type") == "text":
                                dashscope_content.append({"text": item.get("text", "")})
                            elif item.get("type") == "image_url":
                                url = item.get("image_url", {}).get("url", "")
                                if url:
                                    dashscope_content.append({"image": url})
                
                dashscope_msg = {
                    "role": msg["role"],
                    "content": dashscope_content
                }
                dashscope_messages.append(dashscope_msg)
            
            # 调用 Dashscope API
            print(f"[VLMClient-Dashscope] 调用模型 {model_name}，消息数: {len(dashscope_messages)}")
            response = MultiModalConversation.call(
                model=model_name,
                messages=dashscope_messages,
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 2000),  # 增加 max_tokens
                top_p=kwargs.get("top_p", 0.9)
            )
            
            # 检查响应状态
            print(f"[VLMClient-Dashscope] 响应状态码: {response.status_code}")
            if response.status_code == 200:
                output_text = response.output.text if hasattr(response.output, 'text') else str(response.output)
                if not output_text:
                    print(f"[VLMClient-Dashscope] [WARNING] 返回内容为空，尝试降级到OpenAI模式")
                    raise Exception("Dashscope returned empty content")
                print(f"[VLMClient-Dashscope] 成功获得响应，长度: {len(output_text)}")
                return SimpleResponse(output_text)
            else:
                error_msg = getattr(response, 'message', 'Unknown error')
                print(f"[VLMClient-Dashscope] API错误: {error_msg}")
                raise Exception(f"Dashscope API error: {error_msg}")
        
        except Exception as e:
            print(f"[VLMClient] Dashscope 调用失败，降级到 OpenAI 兼容模式: {e}")
            # 降级到 OpenAI 模式
            self.client_type = "openai"
            self.openai_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            return self._create_openai(messages, model_name=model_name, **kwargs)
    
    
    def _create_openai(self, messages: List[Dict[str, Any]], model_name: str, **kwargs) -> Any:
        """使用 OpenAI 兼容接口，带重试机制"""
        max_retries = 3
        retry_delay = 2  # 秒
        
        for attempt in range(max_retries):
            try:
                print(f"[VLMClient-OpenAI] 调用模型 {model_name}，基础URL: {self.base_url}")
                print(f"[VLMClient-OpenAI] 消息数: {len(messages)}")
                print(f"[VLMClient-OpenAI] 尝试 {attempt + 1}/{max_retries}")
                
                response = self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=kwargs.get("temperature", 0.1),
                    max_tokens=kwargs.get("max_tokens", 2000),  # 增加 max_tokens
                    timeout=30,  # 设置超时
                    **{k: v for k, v in kwargs.items() 
                       if k not in ["temperature", "max_tokens"]}
                )
                
                if not response or not response.choices:
                    if attempt < max_retries - 1:
                        print(f"[VLMClient-OpenAI] [WARNING] 响应为空，{retry_delay}秒后重试...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"[VLMClient-OpenAI] [ERROR] 响应为空，已达到最大重试次数")
                        return SimpleResponse("")
                
                if not response.choices[0].message.content:
                    if attempt < max_retries - 1:
                        print(f"[VLMClient-OpenAI] [WARNING] 返回内容为空，{retry_delay}秒后重试...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"[VLMClient-OpenAI] [ERROR] 返回内容为空，已达到最大重试次数")
                        return SimpleResponse("")
                
                content = response.choices[0].message.content
                print(f"[VLMClient-OpenAI] [OK] 成功获得响应，长度: {len(content)} 字符")
                return response
                
            except Exception as e:
                print(f"[VLMClient-OpenAI] [ERROR] API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    print(f"[VLMClient-OpenAI] {retry_delay}秒后重试...")
                    time.sleep(retry_delay)
                else:
                    print(f"[VLMClient-OpenAI] [ERROR] 已达到最大重试次数，放弃")
                    raise
