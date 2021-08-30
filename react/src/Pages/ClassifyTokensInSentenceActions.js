import * as url from "../Common/Url";

export function classifyTokensInSentence(sentence, onSuccess) {
    let isOk = false;
    return function (dispatch, getState) {
        const headers = new Headers()
        headers.append("Accept", "application/json");
        headers.append("Content-Type", "application/json; charset=utf-8");
        fetch(url.CLAFFIFY_TOKENS, {
            method: "post",
            headers,
            body: JSON.stringify({sentence: sentence}),
            credentials: "include",
        }).then((response) => {
            isOk = response.ok
            return response.text()
        }).then((text) => {
            if (isOk) {
                if (onSuccess) onSuccess(text)
            }
        }).catch((error) => {
            console.error(error)
        })
    }
}