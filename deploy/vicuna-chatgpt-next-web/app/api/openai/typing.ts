import type {
  CreateChatCompletionRequest,
  CreateChatCompletionResponse,
} from "openai";

export type ChatRequest = CreateChatCompletionRequest;
export type ChatResponse = CreateChatCompletionResponse;
export type vicunaChatRequest = {
  pic_url: string;
  model: string;
  prompt: string;
  temperature: number;
  max_new_tokens: number | null;
  stop: string;
};
