const cp = getContextPath();

export const CLAFFIFY_TOKENS = cp + "classify_tokens";

export function getContextPath() {
    return (
        window.location.pathname.substring(0, window.location.pathname.lastIndexOf("/")) + "/"
    );
}
