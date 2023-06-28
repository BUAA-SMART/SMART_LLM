import type {
  CreateChatCompletionRequest,
  CreateChatCompletionResponse,
} from "openai";

export type ChatRequest = CreateChatCompletionRequest;
export type ChatResponse = CreateChatCompletionResponse;
export type vicunaChatRequest = {
  model: string;
  prompt: string;
  temperature: number;
  max_new_tokens: number | null;
  stop: string;
};
