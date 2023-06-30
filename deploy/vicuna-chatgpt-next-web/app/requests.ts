import type { ChatResponse, vicunaChatRequest } from "./api/openai/typing";
import { showToast } from "./components/ui-lib";
import { Message, ModelConfig, useAccessStore, useChatStore } from "./store";
const TIME_OUT_MS = 60000;
const makeRequestParam = (
  messages: Message[],
  url:string,
  options?: {
    filterBot?: boolean;
    stream?: boolean;
  },
): vicunaChatRequest => {
  //转化为对象包括角色、内容
  let sendMessages = messages.map((v) => ({
    role: v.role,
    content: v.content,
  }));

  //   pload = {
  //   "model": model_name,#(对应openai v1/completions接口model参数)
  //   "prompt": prompt,#(对应openai v1/completions接口message参数)
  //   "temperature": float(temperature),#(对应openai v1/completions接口temperature参数)
  //   "max_new_tokens": int(max_new_tokens),#(对应openai v1/completions接口max_tokens参数)
  //   "stop": state.sep if state.sep_style == SeparatorStyle.SINGLE else state.sep2,#(对应openai v1/completions接口stop参数)
  // }

  //重新组合为vicuna可以识别的prompt
  //选择是否过滤掉机器人信息
  if (options?.filterBot) {
    sendMessages = sendMessages.filter((m) => m.role !== "assistant");
  }
  //生成vicuna格式的prompt
  const sep = "###"; //采用v1的config
  let prompts = getPrompt(sendMessages, sep);
  //console.log(prompt)
  //将当前参数取出
  const modelConfig = { ...useChatStore.getState().config.modelConfig };
  const model = modelConfig.model
  //设置max_tokens对用户没有太大意义
  // @yidadaa: wont send max_tokens, because it is nonsense for Muggles
  //delete modelConfig.max_tokens;

  //返回vicunaChatRequest格式数据，之后直接用于请求
  //如果是minigpt4，不需要组合为prompt，只需要最后一条信息
  if(model === "minigpt4"){
    prompts = messages[messages.length-1]["content"]
  }
  console.log(prompts)
  return {
    pic_url:url,
    prompt: prompts,
    model: modelConfig.model,
    temperature: modelConfig.temperature,
    max_new_tokens: modelConfig.max_tokens,
    stop: sep,
  };
};

function getPrompt(messages: { role: string; content: string }[], sep: string) {
  //
  let ret = "";
  for (const { role, content } of messages) {
    let newRole = "";
    if (role == "user") {
      newRole = "Human";
    } else {
      newRole = "Assistant";
    }
    if (content) {
      ret += `${newRole}: ${content}${sep}`;
    } else {
      ret += `${newRole}:`;
    }
  }
  ret += `Assistant:`;
  return ret;
}

function getHeaders() {
  const accessStore = useAccessStore.getState();
  let headers: Record<string, string> = {};

  if (accessStore.enabledAccessControl()) {
    headers["access-code"] = accessStore.accessCode;
  }

  if (accessStore.token && accessStore.token.length > 0) {
    headers["token"] = accessStore.token;
  }

  return headers;
}

export function requestOpenaiClient(path: string) {
  return (body: any, method = "POST") =>
    fetch("/api/openai?_vercel_no_cache=1", {
      method,
      headers: {
        "Content-Type": "application/json",
        path,
        ...getHeaders(),
      },
      body: body && JSON.stringify(body),
    });
}

export async function requestChat(messages: Message[]) {
  const req: vicunaChatRequest = makeRequestParam(messages, "",{
    filterBot: true,
  });

  const res = await requestOpenaiClient("v1/chat/completions")(req);

  try {
    const response = (await res.json()) as ChatResponse;
    return response;
  } catch (error) {
    console.error("[Request Chat] ", error, res.body);
  }
}

export async function requestUsage() {
  const formatDate = (d: Date) =>
    `${d.getFullYear()}-${(d.getMonth() + 1).toString().padStart(2, "0")}-${d
      .getDate()
      .toString()
      .padStart(2, "0")}`;
  const ONE_DAY = 2 * 24 * 60 * 60 * 1000;
  const now = new Date(Date.now() + ONE_DAY);
  const startOfMonth = new Date(now.getFullYear(), now.getMonth(), 1);
  const startDate = formatDate(startOfMonth);
  const endDate = formatDate(now);

  const [used, subs] = await Promise.all([
    requestOpenaiClient(
      `dashboard/billing/usage?start_date=${startDate}&end_date=${endDate}`,
    )(null, "GET"),
    requestOpenaiClient("dashboard/billing/subscription")(null, "GET"),
  ]);

  const response = (await used.json()) as {
    total_usage?: number;
    error?: {
      type: string;
      message: string;
    };
  };

  const total = (await subs.json()) as {
    hard_limit_usd?: number;
  };

  if (response.error && response.error.type) {
    showToast(response.error.message);
    return;
  }

  if (response.total_usage) {
    response.total_usage = Math.round(response.total_usage) / 100;
  }

  if (total.hard_limit_usd) {
    total.hard_limit_usd = Math.round(total.hard_limit_usd * 100) / 100;
  }

  return {
    used: response.total_usage,
    subscription: total.hard_limit_usd,
  };
}

export async function requestChatStream(
  messages: Message[],
  url:string,
  options?: {
    filterBot?: boolean;
    modelConfig?: ModelConfig;
    onMessage: (message: string, done: boolean) => void;
    onError: (error: Error, statusCode?: number) => void;
    onController?: (controller: AbortController) => void;
  },
) {
  console.log("requestChatStream");
  console.log(messages)

  //生成request，用于流式请求
  const req = makeRequestParam(messages, url,{
    stream: true,
    filterBot: false,
  });

  const skip_echo_len = req["prompt"].replace("</s>", " ").length + 1;
  console.log("[Request] ", req);

  //设置请求超时时间，定义AbortController 对象并且设置一个超时定时器 setTimeout()，在指定的时间内中止请求
  const controller = new AbortController();
  const reqTimeoutId = setTimeout(() => controller.abort(), TIME_OUT_MS);

  const headers = { "User-Agent": "fastchat Client" };
  const modelConfig = { ...useChatStore.getState().config.modelConfig };
  const model = modelConfig.model
  
  let worker_address = "http://192.168.1.101:21002/worker_generate_stream"
  if(model === "minigpt4"){
    worker_address = "http://192.168.1.101:7789/worker_generate"
  }

  // let worker_address = "/worker/worker_generate_stream"
  // if(model === "minigpt4"){
  //   worker_address = "/worker/worker_generate"
  // }
  //发送请求
  try {
    const res = await fetch(
      worker_address,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(req),
        signal: controller.signal,
      },
    );
    //取消计时器开始，解析流式数据
    clearTimeout(reqTimeoutId);

    let responseText = "";

    //定义finish函数
    const finish = () => {
      options?.onMessage(responseText, true);
      controller.abort();
    };
    
    if (res.ok && model !== "minigpt4") {
      //getReader创建读取器，从响应流中按照块读取数据并将其解码为字符串
      const reader = res.body?.getReader();
      //使用TextDecoder解码
      const decoder = new TextDecoder();

      options?.onController?.(controller);

      //使用while循环不断读取流中数据
      while (true) {
        //每次循环都重新设置一个时限
        const resTimeoutId = setTimeout(() => finish(), TIME_OUT_MS);
        //读取流中数据
        const content = await reader?.read();
        clearTimeout(resTimeoutId);

        if (!content || !content.value) {
          break;
        }

        //流中数据有效则解码，附加到responseText 如何取出text字段，用JSON.PARSE也不行，什么鬼？？
        const text = decoder.decode(content.value, { stream: true });
        //console.log(text);
        const json = parseJson(text);
        if (json.error_code === 0) {
          const output: string = json.text.slice(skip_echo_len).trim();
          responseText = output;
        } else {
          const erroroutput: string = json.text.slice(skip_echo_len).trim();
          responseText = erroroutput + ` ErrorCode: ${json.error_code}`;
          break;
        }

        //将 responseText 作为参数调用 onMessage 回调函数，以通知调用方正在读取新的数据
        const done = content.done;
        options?.onMessage(responseText, false);

        if (done) {
          console.log(responseText); //test
          break;
        }
      }
      //结束，再调用 onMessage 回调函数一次，以便通知调用方已经读取完了所有数据。然后，代码会调用 controller.abort()，请求。
      //onmessgae接受信息进行存储以及显示
      finish();
    } else if (res.ok && model === "minigpt4") {
      const json = await res.json();
      console.log(json);
      if (json.error_code === 0) {
        let output: string = json.text.trim();
        output = output.replace(/<\/Img>|<\/s>|<Img>|<s>/g, '');
        console.log(output);
        options?.onMessage(output, true);
      } else {
        let erroroutput: string = json.text.trim();
        erroroutput = erroroutput.replace(/<\/Img>|<\/s>|<Img>|<s>|<\/img>|<img>/g, '');
        options?.onMessage(erroroutput + ` ErrorCode: ${json.error_code}`, true);
      }
    }else if (res.status === 401) {
      //401处理
      console.error("Unauthorized");
      options?.onError(new Error("Unauthorized"), res.status);
    } else {
      //其他错误
      console.error("Stream Error", res.body);
      options?.onError(new Error("Stream Error"), res.status);
    }
  } catch (err) {
    //err处理
    console.error("NetWork Error", err);
    options?.onError(err as Error);
  }
}

export async function requestWithPrompt(messages: Message[], prompt: string) {
  console.log("requestWithPrompt");
  messages = messages.concat([
    {
      role: "user",
      content: prompt,
      date: new Date().toLocaleString(),
    },
  ]);

  const res = await requestChat(messages);

  return res?.choices?.at(0)?.message?.content ?? "";
}

// To store message streaming controller
export const ControllerPool = {
  controllers: {} as Record<string, AbortController>,

  addController(
    sessionIndex: number,
    messageId: number,
    controller: AbortController,
  ) {
    const key = this.key(sessionIndex, messageId);
    this.controllers[key] = controller;
    return key;
  },

  stop(sessionIndex: number, messageId: number) {
    const key = this.key(sessionIndex, messageId);
    const controller = this.controllers[key];
    controller?.abort();
  },

  stopAll() {
    Object.values(this.controllers).forEach((v) => v.abort());
  },

  hasPending() {
    return Object.values(this.controllers).length > 0;
  },

  remove(sessionIndex: number, messageId: number) {
    const key = this.key(sessionIndex, messageId);
    delete this.controllers[key];
  },

  key(sessionIndex: number, messageIndex: number) {
    return `${sessionIndex},${messageIndex}`;
  },
};

function parseJson(jsonStr: string) {
  // 将 NUL 字符替换为空格
  let decodedStr = jsonStr.replace(/\0/g, " ");
  if (decodedStr.indexOf("%") !== -1) {
    decodedStr = decodedStr.replace(/%([A-Fa-f0-9]{2})/g, (match, p1) => {
      // 判断是否需要解码，如果需要则进行解码
      const code = parseInt(p1, 16);
      return code === 0 ? "\0" : String.fromCharCode(code);
    });
  }
  // 解析 JSON 字符串
  return JSON.parse(decodedStr);
}
